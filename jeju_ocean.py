"""
제주도 DEM + 물결치는 바다 + 하늘을 움직이는 태양  (Blender 스크립트)

[주의] 실행하면 지금 열려 있는 Blender 장면을 새로 시작해요. (저장하지 않은 내용은 사라져요)

[실행 방법]
1) Blender를 열고 상단의 Scripting 탭 → 이 파일을 열고 ▶ Run Script
2) 또는 터미널에서:
     blender --python jeju_ocean.py
     blender --python jeju_ocean.py -- --dem C:\\data\\jeju.asc
     blender -b --python jeju_ocean.py -- --frames 1,120,240 --out C:\\temp\\jeju   (화면 없이 그림 저장)

[DEM 파일]  DEM_PATH(아래 설정) 또는 --dem 으로 지정해요.
  - .asc  : ESRI ASCII Grid (국토지리정보원, SRTM 등에서 변환해서 받을 수 있어요) - 높이는 미터
  - .tif/.png : 높이맵 이미지. 흰색 = HEIGHT_MAX_M(한라산 1947m)로 계산해요.
                (32비트 실수 TIFF에 미터 값이 그대로 들어 있으면 그 값을 그대로 써요)
  - .npy  : numpy 2차원 배열 (미터)
  - 지정하지 않으면 한라산과 오름이 있는 '제주도 모양의 가짜 지형'을 만들어 써요.

[장면 구성]
  🌋 지형  : DEM 높이로 만든 메쉬 (높이를 EXAG배 과장)
  🌊 바다  : Ocean 모디파이어(실제 파도 공식) + 잔물결 범프, 프레임마다 물결이 움직여요
  ☀️ 태양  : 하늘에서 동쪽→서쪽으로 이동 (보이는 태양 + 같은 위치의 Sun 조명)
             태양이 낮을수록 주황색, 높을수록 흰색. 바다에 햇빛이 반짝이며 비쳐요.
"""
import bpy
import bmesh
import math
import os
import sys
import argparse
import numpy as np
from mathutils import Vector

# ======================= 설정 (여기만 바꾸면 돼요) =======================
DEM_PATH = ""                 # 예: r"C:\data\jeju.asc"  (비우면 가짜 지형)
HEIGHT_MAX_M = 1947.0         # 이미지 높이맵일 때 흰색이 뜻하는 높이(m) = 한라산
IMAGE_EXTENT_KM = (90.0, 50.0)  # 이미지/npy 높이맵이 덮는 실제 크기 (동서, 남북) km
EXAG = 4.0                    # 높이 과장 배율 (1이면 실제 비율, 크면 산이 뾰족해져요)
MAX_GRID = 400                # 지형 격자 최대 크기 (클수록 정밀하지만 느려요)

SEA_EXTENT = 170.0            # 바다 한 변의 길이 (1 단위 = 1 km)
OCEAN_RES = 10                # 파도 해상도 (7=빠름 / 10=보통 / 12=고화질, 무거워요)
WAVE_HEIGHT = 0.35            # 파도 높이(대략, 단위 km) - 크면 파도가 거칠어요
WIND = 22.0                   # 바람 세기 (클수록 큰 파도)
CHOPPY = 1.6                  # 파도 뾰족함

FRAME_START, FRAME_END, FPS = 1, 240, 24
SUN_AZ_FROM, SUN_AZ_TO = 40.0, -40.0   # 태양 방향 (북쪽 기준 도, +동쪽 / -서쪽): 오른쪽→왼쪽
SUN_EL_MIN, SUN_EL_MAX = 7.0, 24.0     # 태양 고도 (수평선 위 각도)

CAM_LOC = (0.0, -62.0, 5.0)   # 카메라 위치 (섬의 남쪽 바다에서 북쪽을 봐요) - 낮을수록 바다가 많이 보여요
CAM_PITCH = 5.0               # 카메라가 위로 올려보는 각도
CAM_LENS = 24.0               # 렌즈(mm) - 작을수록 넓게 보여요
# ========================================================================


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--dem", default=DEM_PATH)
    p.add_argument("--frames", default="", help="그림으로 저장할 프레임 번호들 (예: 1,120,240)")
    p.add_argument("--out", default="", help="그림/파일 저장 폴더")
    p.add_argument("--quick", action="store_true", help="빠른 미리보기 화질")
    p.add_argument("--save", default="", help="완성된 .blend 파일 저장 경로")
    return p.parse_args(argv)


def enable_nodes(datablock):
    """재질/월드를 노드 방식으로. (Blender 5.0 이상은 항상 노드라서 따로 켤 필요가 없어요)"""
    if bpy.app.version < (5, 0, 0):
        datablock.use_nodes = True


# ----------------------------- DEM 읽기 -----------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def synthetic_jeju(cell_km=0.25):
    """DEM이 없을 때 쓰는 제주도 모양 지형 (동서 73km x 남북 31km, 한라산 1947m)."""
    nx, ny = int(90 / cell_km), int(50 / cell_km)
    xs = (np.arange(nx) - nx / 2) * cell_km
    ys = (ny / 2 - np.arange(ny)) * cell_km
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(7)

    # 섬의 윤곽을 살짝 울퉁불퉁하게
    ang = np.arctan2(Y / 15.5, X / 36.5)
    wob = (0.04 * np.sin(3 * ang + 1.3) + 0.03 * np.sin(5 * ang + 0.4) + 0.02 * np.sin(9 * ang + 2.1))
    r = np.sqrt((X / 36.5) ** 2 + (Y / 15.5) ** 2) + wob
    p = np.clip(1.0 - r, 0.0, 1.0)                    # 해안 0 → 중심 1

    # 완만한 방패 모양 + 둥근 한라산 봉우리
    peak = np.exp(-(((X + 3.0) ** 2) / (2 * 5.5 ** 2) + (Y ** 2) / (2 * 4.5 ** 2)))
    h = 420.0 * smoothstep(0.0, 0.7, p) + 650.0 * smoothstep(0.3, 1.0, p) ** 2 + 900.0 * peak

    # 오름(작은 화산 언덕)
    for _ in range(110):
        cx, cy = rng.uniform(-30, 30), rng.uniform(-11, 11)
        if ((cx / 36.5) ** 2 + (cy / 15.5) ** 2) > 0.85:
            continue
        amp, rad = rng.uniform(40, 130), rng.uniform(0.4, 1.0)
        h += amp * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rad ** 2)))

    h = np.where(p > 0.0, h, 0.0)
    h = h * (1947.0 / h.max())                              # 가장 높은 곳이 한라산(1947m)이 되도록
    return h.astype(np.float32), cell_km, cell_km


def load_dem(path):
    """(높이 2차원 배열[m], 가로 한 칸 km, 세로 한 칸 km) - 0행이 북쪽."""
    if not path:
        print("[DEM] 파일이 없어서 제주도 모양 가짜 지형을 만듭니다.")
        return synthetic_jeju()
    if not os.path.exists(path):
        raise FileNotFoundError("DEM 파일을 찾을 수 없어요: " + path)

    ext = os.path.splitext(path)[1].lower()
    print("[DEM] 읽는 중:", path)
    if ext == ".asc":
        head = {}
        with open(path, "r") as f:
            for _ in range(6):
                k, v = f.readline().split()
                head[k.lower()] = float(v)
        arr = np.loadtxt(path, skiprows=6, dtype=np.float32)
        arr[arr <= head.get("nodata_value", -9999) + 1e-3] = 0.0
        cs = head["cellsize"]
        if cs < 1.0:   # 경위도(도) 단위
            lat = head.get("yllcorner", 33.4) + arr.shape[0] * cs / 2
            return arr, cs * 111.32 * math.cos(math.radians(lat)), cs * 110.57
        return arr, cs / 1000.0, cs / 1000.0           # 미터 단위(투영좌표)
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        img = bpy.data.images.load(path)
        w, h = img.size
        px = np.empty(w * h * img.channels, dtype=np.float32)
        img.pixels.foreach_get(px)
        arr = np.flipud(px.reshape(h, w, img.channels)[:, :, 0])
        if arr.max() <= 1.5:                            # 0~1 로 정규화된 높이맵
            arr = arr * HEIGHT_MAX_M
    arr[arr < 0] = 0.0
    return arr, IMAGE_EXTENT_KM[0] / arr.shape[1], IMAGE_EXTENT_KM[1] / arr.shape[0]


# ----------------------------- 장면 만들기 -----------------------------
def new_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = FRAME_START, FRAME_END
    sc.render.fps = FPS
    return sc


def make_terrain(heights, dx, dy):
    step = max(1, int(math.ceil(max(heights.shape) / MAX_GRID)))
    h = heights[::step, ::step]
    ny, nx = h.shape
    dx, dy = dx * step, dy * step
    print("[지형] 격자 %d x %d, 한 칸 %.2f km, 최고 %.0f m" % (nx, ny, dx, h.max()))

    # 높이(km) = 미터 / 1000 * 과장. 바닷속은 SEA_FLOOR까지 부드럽게 내려가요(해안선이 자연스럽게).
    z = h / 1000.0 * EXAG
    z = z - 0.5 * (1.0 - np.clip(h / 30.0, 0.0, 1.0))
    xs = (np.arange(nx) - nx / 2.0) * dx
    ys = (ny / 2.0 - np.arange(ny)) * dy
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X, Y, z], axis=-1).reshape(-1, 3)

    idx = np.arange(nx * ny).reshape(ny, nx)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)

    mesh = bpy.data.meshes.new("Jeju_DEM")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    try:
        mesh.shade_smooth()
    except AttributeError:
        for poly in mesh.polygons:
            poly.use_smooth = True
    obj = bpy.data.objects.new("Jeju", mesh)
    bpy.context.collection.objects.link(obj)

    # 높이에 따라 색이 변하는 재질 (모래 → 초록 → 진초록 → 바위)
    zmax = max(float(z.max()), 0.1)
    mat = bpy.data.materials.new("Terrain")
    enable_nodes(mat)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = zmax
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, (0.76, 0.70, 0.50, 1)     # 모래
    cr.elements[1].position, cr.elements[1].color = 1.0, (0.55, 0.52, 0.50, 1)     # 바위
    for pos, col in ((0.03, (0.18, 0.42, 0.14, 1)), (0.20, (0.10, 0.30, 0.10, 1)), (0.60, (0.30, 0.26, 0.20, 1))):
        e = cr.elements.new(pos)
        e.color = col
    nt.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj, (nx * dx, ny * dy)


def make_plane(name, size, location):
    """평면 하나 만들기. (화면이 막 열리는 중에도 되도록 bpy.ops 대신 bpy.data로 만들어요)"""
    h = size / 2.0
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def make_ocean():
    sea = make_plane("Ocean", 1.0, (0, 0, 0))
    mod = sea.modifiers.new("Ocean", "OCEAN")
    mod.geometry_mode = "GENERATE"
    mod.resolution = OCEAN_RES
    mod.viewport_resolution = min(OCEAN_RES, 7)
    mod.spatial_size = 120
    mod.wave_scale = WAVE_HEIGHT
    mod.wave_scale_min = 0.0
    mod.choppiness = CHOPPY
    mod.wind_velocity = WIND
    mod.wave_alignment = 0.3
    mod.damping = 0.5
    mod.depth = 200.0
    mod.random_seed = 3
    mod.use_normals = True
    mod.repeat_x = mod.repeat_y = 1

    # 프레임이 지나면 물결이 움직이도록 time을 드라이버로 연결
    drv = mod.driver_add("time").driver
    drv.type = "SCRIPTED"
    drv.expression = "frame / %d * 1.1" % FPS

    # 생성된 파도 면의 크기를 재서 SEA_EXTENT에 맞추고 섬 가운데로 옮겨요
    bpy.context.view_layer.update()
    ev = sea.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bb = [Vector(c) for c in ev.bound_box]
    minv = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    maxv = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    extent = max(maxv.x - minv.x, maxv.y - minv.y)
    s = SEA_EXTENT / extent
    sea.scale = (s, s, 1.0)
    sea.location = (-(minv.x + maxv.x) / 2 * s, -(minv.y + maxv.y) / 2 * s, 0.0)
    print("[바다] 파도 면 %.1f → %.1f km, 파도 높이 범위 %.2f ~ %.2f" % (extent, SEA_EXTENT, minv.z, maxv.z))

    # 바다 재질: 짙은 청록 + 반짝이는 거울면 + 움직이는 잔물결 범프
    mat = bpy.data.materials.new("Sea")
    enable_nodes(mat)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.01, 0.12, 0.20, 1)
    bsdf.inputs["Roughness"].default_value = 0.04
    bsdf.inputs["Metallic"].default_value = 0.0
    # 월드 좌표 기준 무늬 → 파도 면과 바깥 평평한 바다가 이음새 없이 이어져요
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    # (1) 방향이 있는 물결 줄무늬 (파도마루가 줄지어 밀려오는 느낌)
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.wave_profile = "SIN"
    wave.inputs["Scale"].default_value = 2.2
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 3.0
    wave.inputs["Detail Scale"].default_value = 1.8
    # (2) 잔잔한 잔물결 (4D 노이즈: 시간에 따라 변해요)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 14.0
    noise.inputs["Detail"].default_value = 8.0
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.65
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = 0.45
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.08
    nt.links.new(geo.outputs["Position"], wave.inputs["Vector"])
    nt.links.new(geo.outputs["Position"], noise.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], mul.inputs[0])
    nt.links.new(noise.outputs["Fac"], add.inputs[0])
    nt.links.new(mul.outputs["Value"], add.inputs[2])
    nt.links.new(add.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    sea.data.materials.append(mat)

    # 파도 면 바깥은 수평선까지 평평한 바다가 이어지도록 (같은 재질의 아주 큰 판)
    far = make_plane("OceanFar", 6000.0, (0, 0, -0.03))
    far.data.materials.append(mat)

    # 잔물결 무늬도 프레임에 따라 흘러가게 (4D 노이즈의 W 값에 드라이버)
    w_drv = noise.inputs["W"].driver_add("default_value").driver
    w_drv.type = "SCRIPTED"
    w_drv.expression = "frame / 40"
    p_drv = wave.inputs["Phase Offset"].driver_add("default_value").driver
    p_drv.type = "SCRIPTED"
    p_drv.expression = "frame / 12"          # 줄무늬가 서서히 밀려와요
    return sea


def lerp(a, b, t):
    return a + (b - a) * t


def make_sun():
    """보이는 태양(구) + 같은 위치의 Sun 조명. 하늘의 호를 따라 움직여요."""
    R = 420.0                                           # 태양까지의 거리
    target = bpy.data.objects.new("SunTarget", None)
    bpy.context.collection.objects.link(target)

    # 보이는 태양 (빛나는 구)
    disc_mesh = bpy.data.meshes.new("SunDisc")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=R * math.tan(math.radians(2.2)))
    bm.to_mesh(disc_mesh)
    bm.free()
    try:
        disc_mesh.shade_smooth()
    except AttributeError:
        pass
    disc = bpy.data.objects.new("SunDisc", disc_mesh)
    disc.location = (0, R, 0)
    bpy.context.collection.objects.link(disc)
    disc.visible_shadow = False
    mat = bpy.data.materials.new("SunGlow")
    enable_nodes(mat)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 60.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    disc.data.materials.append(mat)

    # 실제로 빛을 내는 Sun 조명 (항상 섬 쪽을 바라봐요)
    ld = bpy.data.lights.new("SunLight", "SUN")
    ld.energy = 4.0
    ld.angle = math.radians(1.2)
    lamp = bpy.data.objects.new("SunLight", ld)
    bpy.context.collection.objects.link(lamp)
    tr = lamp.constraints.new("TRACK_TO")
    tr.target = target
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"

    # 프레임마다 위치/색을 기록 (여러 개의 키프레임 → 부드럽게 이동)
    n = 24
    for k in range(n + 1):
        t = k / n
        frame = round(lerp(FRAME_START, FRAME_END, t))
        az = math.radians(lerp(SUN_AZ_FROM, SUN_AZ_TO, t))
        el_norm = math.sin(math.pi * t)                 # 0 → 1 → 0
        el = math.radians(lerp(SUN_EL_MIN, SUN_EL_MAX, el_norm))
        pos = Vector((R * math.sin(az) * math.cos(el), R * math.cos(az) * math.cos(el), R * math.sin(el)))
        # 낮은 태양 = 주황, 높은 태양 = 흰빛
        col = (1.0, lerp(0.50, 0.93, el_norm), lerp(0.22, 0.82, el_norm))

        disc.location = pos
        disc.keyframe_insert("location", frame=frame)
        lamp.location = pos
        lamp.keyframe_insert("location", frame=frame)
        ld.color = col
        ld.keyframe_insert("color", frame=frame)
        em.inputs["Color"].default_value = (*col, 1.0)
        em.inputs["Color"].keyframe_insert("default_value", frame=frame)
    return disc, lamp


def make_world():
    """하늘색 (아래는 수평선에 가까운 밝은 색, 위는 진한 파랑)."""
    world = bpy.data.worlds.new("Sky")
    enable_nodes(world)
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")          # 바라보는 방향의 높이(-1~1) → 0~1
    mr.clamp = True
    mr.inputs["From Min"].default_value = -0.05
    mr.inputs["From Max"].default_value = 0.85
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, (0.75, 0.85, 0.95, 1)   # 수평선: 밝은 하늘색
    cr.elements[1].position, cr.elements[1].color = 1.0, (0.06, 0.22, 0.62, 1)   # 머리 위: 진한 파랑
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world


def make_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = CAM_LENS
    cam_data.clip_end = 3000.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = CAM_LOC
    cam.rotation_euler = (math.radians(90.0 + CAM_PITCH), 0.0, 0.0)   # 북쪽(+Y)을 바라봐요
    bpy.context.scene.camera = cam
    return cam


def setup_render(sc, quick):
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24 if quick else 96
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 960 if quick else 1920
    sc.render.resolution_y = 540 if quick else 1080
    sc.render.image_settings.file_format = "PNG"


def show_in_viewport():
    """화면이 있는 Blender에서 카메라 시점 + 재질 미리보기로 바꾸고, 애니메이션을 자동 재생해요."""
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            try:
                space.shading.type = "MATERIAL"              # 부드럽게 재생되는 미리보기
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = True
                space.overlay.show_overlays = False          # 격자/조명 표시는 숨겨요
            except Exception as e:
                print("[보기 설정 일부 실패]", e)
            space.region_3d.view_perspective = "CAMERA"       # 카메라 시점
            with bpy.context.temp_override(window=win, area=area):
                if not bpy.context.screen.is_animation_playing:
                    bpy.ops.screen.animation_play()           # 재생 (Space와 같아요)
            print("[보기] 카메라 시점으로 자동 재생 중 (멈추기: Space)")
            return None
    return 0.5                                                # 화면이 아직 준비 중이면 다시 시도


def main():
    args = parse_args()
    sc = new_scene()
    heights, dx, dy = load_dem(args.dem)
    make_terrain(heights, dx, dy)
    make_ocean()
    make_world()
    make_sun()
    make_camera()
    setup_render(sc, args.quick)

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), "jeju_render")
    os.makedirs(out_dir, exist_ok=True)

    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)
        print("[저장] .blend:", args.save)

    if args.frames:                                    # 지정한 프레임만 그림으로 저장
        for f in [int(x) for x in args.frames.split(",") if x.strip()]:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(out_dir, "jeju_%04d.png" % f)
            bpy.ops.render.render(write_still=True)
            print("[렌더]", sc.render.filepath)
    else:
        sc.frame_set(FRAME_START)
        print("[완료] 재생(Space)하면 물결과 태양이 움직여요. 렌더링은 Render → Render Animation (Ctrl+F12)")
        if not bpy.app.background:
            bpy.app.timers.register(show_in_viewport, first_interval=1.5)


main()

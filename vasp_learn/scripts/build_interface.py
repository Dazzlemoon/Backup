import os
from pymatgen.core import Structure
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder

# ================= 设置区 =================
# 指向你刚才下载并保留的纯相文件路径
# 注意：这里使用的是 VASP 格式的文件，路径要对
elec_path = "Na_Na3SbS4_Data/structures/Na3SbS4_mp-10167.vasp" 
anode_path = "Na_Na3SbS4_Data/structures/Na_mp-127.vasp"

output_dir = "Na_Na3SbS4_Data/Interface_structures"
os.makedirs(output_dir, exist_ok=True)
# =========================================

def main():
    print("1. 读取结构...")
    try:
        sub_s = Structure.from_file(elec_path) # 电解质作为基底
        film_s = Structure.from_file(anode_path) # Na 作为薄膜
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print("2. 寻找晶格匹配 (这可能需要几分钟)...")
    # ZSL 算法寻找两边晶格的最小公倍数
    zsl = ZSLGenerator(
        max_area_ratio_tol=0.09,  # 允许 9% 的面积差异
        max_area=400,             # 最大界面面积 (太大计算跑不动)
        max_l_by_a_per_tol=0.05,  # 边长容差
        max_angle_tol=0.01        # 角度容差
    )

    # 构建器：尝试让 (001) 面接触 (001) 面
    builder = CoherentInterfaceBuilder(
        substrate_structure=sub_s,
        film_structure=film_s,
        film_miller=(0, 0, 1), 
        substrate_miller=(0, 0, 1),
        zslgen=zsl
    )

    print("3. 生成界面构型...")
    # 获取前 2 个最佳匹配
    interfaces = list(builder.get_interfaces(termination=("Na", "S")))[:2]

    if not interfaces:
        print("未找到合适的匹配！请尝试放宽 ZSLGenerator 的参数。")
        return

    for i, iface in enumerate(interfaces):
        # 这里的关键：设置两边的厚度
        # film_thickness=10 (Na层厚度 10埃)
        # substrate_thickness=12 (电解质层厚度 12埃)
        # in_plane_scaling=1 (不强制拉伸，保持晶格匹配)
        s = iface.get_structure(
            in_plane_scaling=1, 
            film_thickness=10, 
            substrate_thickness=12
        )
        
        # 微扰一下，打破对称性，防止卡在鞍点
        s.perturb(0.1)
        
        # 转换为 VASP 格式并保存
        filename = os.path.join(output_dir, f"Interface_Config_{i}.vasp")
        s.to(filename=filename, fmt="poscar")
        
        print(f"--- 成功生成第 {i} 个界面 ---")
        print(f"  原子数: {len(s)}")
        print(f"  保存路径: {filename}")
        
        # 安全检查：如果原子数超过 300，可能太慢了
        if len(s) > 300:
            print("  [警告] 原子数过多，计算会非常慢！")

if __name__ == "__main__":
    main()
from aumgo import doped_vs_undoped

# reference values from paper
d = -2.7
u = 929
print(f"Reference: (E_wetting - E_non_wetting: Doped {d:.3f}meV, Undoped {u:.3f}meV)")

for key, (d, u) in doped_vs_undoped.items():
    # physical models have the correct hierarchy of wetting vs non-wetting
    good = (d < u) and d < 0

    if good:
        mark = "✅"
    else:
        mark = "❌"

    print(
        f"{key}: {mark} (E_wetting - E_non_wetting: Doped {d:.3f}meV, Undoped {u:.3f}meV)"
    )

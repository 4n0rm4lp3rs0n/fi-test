from selfmade.fi import pipeline, get_latest_record
from selfmade.genome import vanilla_run, guided_run
from pathlib import Path
import json

experiments = 5
generations = 10
population_size = 1000
edge_limit = 9
layer_limit = 7

conv1x1 = 'conv1x1-bn-relu'
conv3x3 = 'conv3x3-bn-relu'
maxpool3x3 = 'maxpool3x3'

operations = [conv1x1, conv3x3, maxpool3x3]

record_root = Path("./records_v")

# vadir = record_root / "vanilla"

for e in range(experiments):
    print(f"experiment no.{e}")
    vadir = record_root / f"exp{e:03d}"

    vanilla_dict = vanilla_run(operations, population_size, edge_limit, layer_limit, generations)

    vadf = vanilla_dict["population_data"]
    vacfg = vanilla_dict["configs"]

    vadir.mkdir(parents=True, exist_ok=True)

    vanilla_res = pipeline(data=vadf, device='cpu')

    print("vanilla correlation: ", vanilla_res["corr"])
    print("vanilla p-value: ", vanilla_res["p_value"])

    vadf.to_csv(vadir / "population.csv")
    vanilla_res["df_importance"].to_csv(vadir / "fi.csv")
    vanilla_res["bit_directions"].to_csv(vadir / "bitgui.csv")
    vanilla_res["layer_report"].to_csv(vadir / "layergui.csv")

    vacfg["r2"] = vanilla_res["r2"]
    vacfg["mae"] = vanilla_res["mae"]
    vacfg["corr"] = vanilla_res["corr"]
    vacfg["p_value"] = vanilla_res["p_value"]

    with open(vadir / "config.json", "w") as c:
        json.dump(vacfg, c, indent = 4)
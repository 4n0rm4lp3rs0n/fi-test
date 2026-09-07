from selfmade.fi import pipeline, get_latest_record
from selfmade.genome import vanilla_run, guided_run, get_stats
from pathlib import Path
import json
import pandas as pd

# experiments = 0 means only vanilla
experiments = 5
generations = 10
population_size = 1000
mode = "nasbench101"

stats = get_stats(mode)

edge_limit = stats["edge_limit"]
layer_limit = stats["layer_limit"]
operations = stats["operations"]

record_root = Path("./records_g")

vadir = record_root / "vanilla"

if not vadir.exists() or not any(vadir.iterdir()):

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

if experiments == 0:
    exit()
    
gui_format = "gui"
gpath = record_root
for i in range(1, experiments + 1):
    files = []
    print(f"experiment no.{i}")
    record_path = get_latest_record(gpath, gui_format, increment=True)
    if not isinstance(record_path, Path):
        # ordered in alphabet, so bitgui, fi, layergui and population
        files = [f for f in vadir.iterdir() if f.is_file()]
        # files = [f for f in vadir.iterdir() if f.is_file() and f.suffix.lower() != ".json"]
    else:
        # files = [f for f in record_path.iterdir() if f.is_file() and f.suffix.lower() != ".json"]
        files = [f for f in record_path.iterdir() if f.is_file()]

    bitgui = pd.read_csv(files[0])
    with open(files[1], "r", encoding="utf-8") as file:
        config = json.load(file)
    feature_importance = pd.read_csv(files[2])
    layergui = pd.read_csv(files[3])
    pop = pd.read_csv(files[4])

    # print("prior data gathered")

    gui_dict = guided_run(operations, population_size, edge_limit, layer_limit, generations, bit_guidance=bitgui,
                        layer_guidance=layergui, feature_importance=feature_importance, config=config, pre_pop=pop)

    guidf = gui_dict["population_data"]
    guicfg = gui_dict["configs"]

    i = 1
    while (gpath / f"{gui_format}{i:03d}").exists():
        i += 1

    guipath = gpath / f"{gui_format}{i:03d}"
    guipath.mkdir(parents=True)

    gui_res = pipeline(data = guidf, device="cpu")
    print("guidance correlation: ", gui_res["corr"])
    print("guidance p-value: ", gui_res["p_value"])

    guicfg["r2"] = gui_res["r2"]
    guicfg["mae"] = gui_res["mae"]
    guicfg["corr"] = gui_res["corr"]
    guicfg["p_value"] = gui_res["p_value"]

    guidf.to_csv(guipath / "population.csv")
    gui_res["df_importance"].to_csv(guipath / "fi.csv")
    gui_res["bit_directions"].to_csv(guipath / "bitgui.csv")
    gui_res["layer_report"].to_csv(guipath / "layergui.csv")

    with open(guipath / "config.json", "w") as c:
        json.dump(guicfg, c, indent=4)
    
import os
from option import args

txt_files = ["CVC-300.txt", "CVC-ClinicDB.txt", "CVC-ColonDB.txt", "ETIS-LaribPolypDB.txt", "Kvasir.txt"]
save_dir = os.path.join(args.exp_dir, args.exp_name, 'eval_fig')

file_weights = {
    "CVC-300.txt": 60,
    "CVC-ClinicDB.txt": 62,
    "Kvasir.txt": 100,
    "CVC-ColonDB.txt": 380,
    "ETIS-LaribPolypDB.txt": 196
}

total_weight = sum(file_weights.values())

results = {}

for file_name, weight in file_weights.items():
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'r') as f:
        next(f)  # jump the first line
        lines = f.readlines()
        for line in lines:
            parts = line.split(", ")
            checkpoint_name = parts[0].split(":")[1].strip()
            mean_dice = float(parts[1].split("is ")[1].strip())
            mean_iou = float(parts[2].split("is ")[1].strip())
            
            if checkpoint_name not in results:
                results[checkpoint_name] = {"meanDice": 0.0, "meanIoU": 0.0}
            
            results[checkpoint_name]["meanDice"] += mean_dice * weight
            results[checkpoint_name]["meanIoU"] += mean_iou * weight

for checkpoint_name, data in results.items():
    data["meanDice"] /= total_weight
    data["meanIoU"] /= total_weight

max_dice_checkpoint = max(results, key=lambda k: results[k]["meanDice"])

output_file = os.path.join(save_dir, "average_results.txt")
with open(output_file, 'w') as f:
    for checkpoint_name, data in results.items():
        f.write(f"model:{checkpoint_name}, mean Dice is {data['meanDice']:.4f}, IoU is {data['meanIoU']:.4f}\n")
    f.write(f"max model:{max_dice_checkpoint}, mean Dice is {results[max_dice_checkpoint]['meanDice']:.4f}, IoU is {results[max_dice_checkpoint]['meanIoU']:.4f}\n")

print(f"the average results is saved in: {output_file}")
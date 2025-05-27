# extract_timestamps.py

input_file = "vid_to_png/slamtest4/data.csv" 
output_file = "vid_to_png/slamtest4/MH01.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    # Skip header line
    next(infile)

    for line in infile:
        timestamp = line.split(',')[0].strip()
        if timestamp:
            outfile.write(timestamp + "\n")

print(f"Timestamps extracted to: {output_file}")
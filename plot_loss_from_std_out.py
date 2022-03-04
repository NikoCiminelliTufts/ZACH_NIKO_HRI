import sys
import matplotlib.pyplot as plt

# read file as argument
file = sys.argv[1]

loss = []
with open(file, "r") as f:
    for line in f.readlines():
        if line[:10] == "training e":
                
            prefix = "total_loss: "
            loss_start_ind = line.index(prefix) + len(prefix)
            
            suffix = "recon_loss:"
            loss_end_ind = line.index(suffix)

            loss.append(float(line[loss_start_ind:loss_end_ind]))

print("collected {} losses".format(len(loss)))

plt.plot(loss)
plt.savefig(file + ".png", format="png")
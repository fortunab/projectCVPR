from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, GRID = "#0072B2", "#D55E00", "#D9DEE5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":9.5,"axes.labelsize":8.5,"xtick.labelsize":8.5,"ytick.labelsize":8,"legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42})
def polish(ax):
    ax.set_ylim(0,100); ax.set_yticks(np.arange(0,101,20)); ax.grid(axis="y",color=GRID,linewidth=.65,zorder=0); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_color("#78828C"); ax.spines["bottom"].set_color("#78828C"); ax.tick_params(axis="both",length=0,pad=4)
def make_plot(path, groups, series_a, series_b, title_a, title_b, legend_b):
    x=np.arange(len(groups)); w=.31; fig, axes=plt.subplots(1,2,figsize=(7.15,2.75),sharey=True)
    for ax, a, b, title in [(axes[0],series_a[0],series_b[0],title_a),(axes[1],series_a[1],series_b[1],title_b)]:
        bars_a=ax.bar(x-w/2,a,w,color=BLUE,label="CBMP",zorder=3); bars_b=ax.bar(x+w/2,b,w,color=ORANGE,label=legend_b,zorder=3)
        ax.set_xticks(x,groups); ax.set_title(title,pad=7,weight="semibold"); ax.set_ylabel("Balanced accuracy (%)" if "polypgen" in path else "Score (%)"); polish(ax)
        for bars in (bars_a,bars_b):
            for bar in bars:
                v=bar.get_height(); ax.annotate(f"{v:.2f}",(bar.get_x()+bar.get_width()/2,v),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=7.3,color="#20252B")
    fig.legend(handles=[axes[0].containers[0][0],axes[0].containers[1][0]],labels=["CBMP",legend_b],loc="upper center",bbox_to_anchor=(.5,1.04),ncol=2,frameon=False,columnspacing=1.5,handlelength=1.0)
    fig.tight_layout(rect=(0,0,1,.93),w_pad=1.8); fig.savefig(OUT/(path+".pdf"),bbox_inches="tight"); fig.savefig(OUT/(path+".png"),dpi=240,bbox_inches="tight"); plt.close(fig)
# From Table tab: PolypGen mean and worst-center BA; best learned comparator per encoder.
make_plot("polypgen_classifier_comparison",["ResNet18","DINOv2"],[[58.35,59.54],[42.14,45.31]],[[70.20,68.17],[57.17,61.45]],"Mean across held-out centers","Worst held-out center","Strongest learned baseline")
# From Table tab: custom CAMELYON patient-capped subset, target hospitals H2 and H1.
make_plot("camelyon_classifier_comparison",["H2 (primary)","H1 (secondary)"],[[85.40,77.58],[92.96,85.35]],[[88.88,80.30],[95.14,91.71]],"Balanced accuracy","AUROC","Linear logistic")
print("Generated PolypGen and CAMELYON figures.")
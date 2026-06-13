import os
import sys

# 确保项目根目录在 sys.path 中 (支持独立运行或作为 tools 子模块导入)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.test_plots import Figdata, show_figs

class AnmScoreFigData():
    def __init__(self):
        self.figdatas = []

    def anm_score_to_figdata(self, scores, title=""):
        nml_scores = [x[1] for x in scores if x[0]==0]
        anm_scores = [x[1] for x in scores if x[0]==1]

        figdata = Figdata(
            data=nml_scores,
            data2=anm_scores,
            type="boxplot",
            labels=["nml","anm"],
            ylabel="score",
            title=title
        )

        return figdata


    def append_figdata(self, figdata):
        self.figdatas.append(figdata)


    def show_fig(self, title="anm_score", export_dir="results", is_display_console=False):
        show_figs(
            *self.figdatas,
            fold_interval=len(self.figdatas),
            sup_title=title,
            export_path=f"{export_dir}/{title}.png",
            is_display_console=is_display_console
        )

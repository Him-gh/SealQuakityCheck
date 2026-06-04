import matplotlib
import torch
import matplotlib.pyplot as plt
from IPython import display

###-------------------------- 工具函数 --------------------------###
def use_svg_display():
    """
    SVG格式
    """
    plt.rcParams['savefig.format'] = 'svg'

def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    axes.grid()


def has_one_axis(X):
    """判断X是否为一维数据"""
    return (hasattr(X, 'ndim') and X.ndim == 1 or 
            isinstance(X, list) and not hasattr(X[0], "__len__"))

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
###-------------------------- 绘制数据点 --------------------------###
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-', 'r:'), figsize=(3.5, 2.5), axes=None,
         show=True, save_path=None):
    """
    绘制数据点

    参数:
    ----------
    show : bool, 默认=True
        是否显示图形窗口
    save_path : str, 默认=None
        如果提供路径，则保存图形到文件
    """
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    # 处理输入数据
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    
    if len(X) != len(Y):
        X = X * len(Y)
    
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    # 保存图形（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg' 
                   if save_path.endswith('.svg') else None)
    # 显示图形（如果需要）
    if show:
        plt.show()
    elif not show and not save_path:
        plt.close()  # 如果不显示也不保存，关闭图形释放内存


###-------------------------- 绘制图像列表 --------------------------###
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    """
    绘制图像列表

    参数:
    ----------
    imgs: 图像列表，支持 numpy array 或 torch.Tensor
    num_rows, num_cols: 子图行列数
    titles: 标题列表（可选）
    scale: 缩放因子
    """
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten()
    for i,(ax,img)in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


###-------------------------- 绘制动画 --------------------------###
class Animator:
    """
    Animator for jupyter notebook

    参数:
    ----------
    xlabel, ylabel : str, 坐标轴标签
    legend : list, 图例标签列表
    xlim, ylim : tuple, 坐标轴范围，如 (0, 10)
    xscale, yscale : str, 坐标轴刻度类型 ('linear', 'log', 'symlog' 等)
    fmts : tuple, 每条曲线的线型和颜色格式
    nrows, ncols : int, 子图的行数和列数（当前版本只使用第一个子图）
    figsize : tuple, 画布大小 (宽, 高) 英寸
    """
    def __init__(self,xlabel=None,ylabel=None,legend=None,xlim=None,
                 ylim=None,xscale='linear',yscale='linear',
                 fmts=('-','m--','g-.','r:'),nrows=1,ncols=1,
                 figsize=(3.5,2.5)):
        if legend is None:
            legend=[]
        use_svg_display()
        self.fig,self.axes=plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols==1:
            self.axes=[self.axes,]
        self.config_axes=lambda: set_axes(
            self.axes[0],xlabel,ylabel,xlim,ylim,xscale,yscale,legend
        )
        self.X,self.Y,self.fmts=None,None,fmts
    def add(self,x,y):
        """
        向图表中添加多个数据点
        
        参数:
        ----------
        x : float or list
            新数据点的 x 坐标，如果是单条曲线可以是数值，多条曲线需要是列表
        y : float or list
            新数据点的 y 坐标，如果是单条曲线可以是数值，多条曲线需要是列表
        """
        if not hasattr(y,"__len__"):
            y=[y]
        n=len(y)
        if not hasattr(x,"__len__"):
            x=[x]*n
        if not self.X:
            self.X=[[] for _ in range(n)]
        if not self.Y:
            self.Y=[[] for _ in range(n)]
        for i ,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y ,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
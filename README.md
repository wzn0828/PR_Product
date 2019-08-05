# PR Product
Code for paper "PR Product:  A Substitute for Inner Product in Neural Networks", containing PR-FC, PR-CNN and PR-LSTM.

The code is implemented based on Pytorch.
# Usage
The usage of PR-X is the same as the one of nn.X:
        
        import PR
        
        pr_fc = PR.PRLinear(100, 200)
        pr_cnn = PR.PRConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        pr_lstmcell = PR.PRLSTMCell(256, 512)

# Citing
If you use 'PR Product' in a scientific publication, we would appreciate references to the following paper:

**PR Product:  A Substitute for Inner Product in Neural Networks**. [arXiv:1904.13148](https://arxiv.org/abs/1904.13148)

Biblatex entry:

        @article{wang2019pr,
        title={PR Product: A Substitute for Inner Product in Neural Networks},
        author={Wang, Zhennan and Zou, Wenbin and Xu, Chen},
        journal={arXiv preprint arXiv:1904.13148},
        year={2019}
        }
# License
This code is released under the MIT License (refer to the LICENSE file for details).
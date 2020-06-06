# PR Product
Code for ICCV2019 Oral paper "PR Product:  A Substitute for Inner Product in Neural Networks", containing PR-FC, PR-CNN and PR-LSTM.

The code is implemented based on Pytorch.
# Usage
The usage of PR-X is the same as the one of nn.X:
        
        import PR
        
        pr_fc = PR.PRLinear(100, 200)
        pr_cnn = PR.PRConv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        pr_lstmcell = PR.PRLSTMCell(256, 512)

# Citing
If you use 'PR Product' in a scientific publication, we would appreciate references to the following paper:

**PR Product:  A Substitute for Inner Product in Neural Networks**. [ICCV2019 Oral: PR Product](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PR_Product_A_Substitute_for_Inner_Product_in_Neural_Networks_ICCV_2019_paper.pdf)

Biblatex entry:

        @inproceedings{wang2019pr,
                       title={PR Product: A Substitute for Inner Product in Neural Networks},
                       author={Wang, Zhennan and Zou, Wenbin and Xu, Chen},
                       booktitle={Proceedings of the IEEE International Conference on Computer Vision},
                       pages={6013--6022},
                       year={2019}
                       }
# License
This code is released under the MIT License (refer to the LICENSE file for details).

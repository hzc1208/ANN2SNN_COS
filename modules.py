from torch import nn
import torch.nn.functional as F
import torch
from spikingjelly.clock_driven import neuron
from torch.autograd import Function

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class IFNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(IFNeuron, self).__init__()
        self.v_threshold = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
        
    def forward(self, x):      
        x = x / self.v_threshold               
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.5)   
        
        self.t += 1
        
        x = self.neuron(x)
                
        return x * self.v_threshold
        
    def reset(self):
        self.t = 0
        self.neuron.reset()


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

qcfs = FloorLayer.apply

class QCFS(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        
    def forward(self, x):
        
        output = x / self.up
        output = qcfs(output*self.t+0.5)
        output = output / self.t
        output = torch.clamp(output, 0, 1)
        output = output * self.up
             
        return output


class MPLayer(nn.Module):
    def __init__(self, v_threshold, presim_len, sim_len, batchsize):
        super().__init__()        
        self.neuron = neuron.IFNode(v_reset=None)
        self.v_threshold = v_threshold
        self.t = 0
        self.presim_len = presim_len
        self.sim_len = sim_len
        self.batchsize = batchsize

        self.snn_rate = None
        self.snn_rate_opt = None
        self.snn_mode = True
        self.act = QCFS(up=v_threshold,t=presim_len)
        self.arr = []
        self.arr2 = []
        self.tot_num2 = 0.
        
 
    def forward(self, x):
          with torch.no_grad():
              if self.snn_mode == True:
                  xx = 0.
                  mem_seq = []
                  mem_seq2 = []
                  output_seq = []
              
                  if self.t == 0:

                      if x.dim() == 4:
                          if x.shape[0] <= self.batchsize:
                              xx = x.unsqueeze(0).repeat(self.sim_len,1,1,1,1)
                          else:
                              xx = x.reshape(self.sim_len,x.shape[0]//self.sim_len,x.shape[1],x.shape[2],x.shape[3])
                      else:
                          xx = x.reshape(self.sim_len,x.shape[0]//self.sim_len,x.shape[1])
                          
                      self.neuron.reset()
                      self.neuron(torch.ones_like(xx[0])*0.5)
                  
                  snn_rate = torch.zeros_like(xx[0])
                  #self.snn_rate_opt = torch.zeros_like(xx[0])
                  init_mem_pot = torch.ones_like(snn_rate)*0.5
                  
                  membrane_lower, membrane_lower_two, membrane_higher_two = 0,0,0
                  membrane_higher_three, membrane_lower_three = 0,0
                  membrane_higher_four, membrane_lower_four = 0,0
                  
                  for opt_epoch in range(1):
                      self.t = 0
                      for t in range(self.presim_len):
                          input_snn = xx[t] /self.v_threshold
                          output = self.neuron(input_snn)              
                          mem = self.neuron.v + 1.            
                          snn_rate += output 
                          self.t += 1
                          
                          if self.t <= self.presim_len:                
                              mem_seq.append(torch.where(output>0,mem,torch.ones_like(output)*1000.))
                              mem_seq2.append(torch.where(output>0,torch.ones_like(output)*(-1000.),self.neuron.v))
    
                      
                      if self.t == self.presim_len and self.presim_len > 0:                     
                          if opt_epoch == 0:
                              membrane_lower = torch.where(self.neuron.v<0,(snn_rate>1e-3).float(),torch.zeros_like(snn_rate))
                              '''
                              membrane_lower_two = torch.where(self.neuron.v<-1.,(snn_rate>1.).float(),torch.zeros_like(snn_rate))
                              membrane_higher_two = (self.neuron.v>=2.)
                              membrane_lower_three = torch.where(self.neuron.v<-2.,(snn_rate>2.).float(),torch.zeros_like(snn_rate))
                              membrane_higher_three = (self.neuron.v>=3.)
                              membrane_lower_four = torch.where(self.neuron.v<-3.,(snn_rate>3.).float(),torch.zeros_like(snn_rate))
                              membrane_higher_four = (self.neuron.v>=4.)
                              '''                         
                          
                          mem_val = torch.topk(torch.stack(mem_seq),1,dim=0,largest=False)[0]                          
                          mem_val2 = torch.topk(torch.stack(mem_seq2),1,dim=0,largest=True)[0]
 
                          val = torch.where((mem_val[0]- 0.999)>1.,(mem_val[0]- 0.999),torch.ones_like(snn_rate))
                          val2 = torch.where((1.001-mem_val2[0])>1.,(1.001-mem_val2[0]),torch.ones_like(snn_rate))
                            

                          if opt_epoch == 0:
                              mem_init = init_mem_pot - membrane_lower * val                       
                              mem_init = torch.where(self.neuron.v>=1.,torch.ones_like(snn_rate)*0.5+val2,mem_init)
                          elif opt_epoch == 1:
                              mem_init = init_mem_pot - membrane_lower_two * val
                              mem_init = mem_init + membrane_higher_two * val2
                          elif opt_epoch == 2:
                              mem_init = init_mem_pot - membrane_lower_three * val
                              mem_init = mem_init + membrane_higher_three * val2                         
                          elif opt_epoch == 3:
                              mem_init = init_mem_pot - membrane_lower_four * val
                              mem_init = mem_init + membrane_higher_four * val2
                          elif opt_epoch == 4:
                              mem_init = init_mem_pot - membrane_lower_five * val
                              mem_init = mem_init + membrane_higher_five * val2
                          elif opt_epoch == 5:
                              mem_init = init_mem_pot - membrane_lower_six * val
                              mem_init = mem_init + membrane_higher_six * val2                         
                          elif opt_epoch == 6:
                              mem_init = init_mem_pot - membrane_lower_seven * val
                              mem_init = mem_init + membrane_higher_seven * val2 
                          else:
                              mem_init = init_mem_pot - membrane_lower_eight * val
                              mem_init = mem_init + membrane_higher_eight * val2                               
                                                                                         
                                            
                          self.neuron.reset()
                          self.neuron.v = mem_init
                          init_mem_pot = mem_init
                          mem_seq = mem_seq2 = []
                  
                  
                  for t in range(self.sim_len):
                      input_snn = xx[t]/self.v_threshold
                      output = self.neuron(input_snn)

                      output_seq.append(output)
                      self.t += 1
                      #if t < self.presim_len:
                          #self.snn_rate_opt += output
                      
                  #self.snn_rate_opt = self.snn_rate_opt / self.presim_len
                      
                  if self.t == self.presim_len + self.sim_len:                     
                      self.t = 0  
                  
                  output_seq = torch.stack(output_seq)
                  
                  if output_seq.dim() == 5:
                      output_seq = output_seq.reshape(output_seq.shape[0]*output_seq.shape[1],output_seq.shape[2],output_seq.shape[3],output_seq.shape[4])
                  else:
                      output_seq = output_seq.reshape(output_seq.shape[0]*output_seq.shape[1],output_seq.shape[2])
                  
                  return output_seq*self.v_threshold
              else:
                  output = self.act(x)
                  tot_num = output.numel()
                  
                  tot_zero = (((output/self.v_threshold - self.snn_rate_opt)*self.presim_len).abs() < 0.01).sum()
                  tot_one = (((output/self.v_threshold - self.snn_rate_opt)*self.presim_len).abs() < 1.01).sum()
                  tot_two = (((output/self.v_threshold - self.snn_rate_opt)*self.presim_len).abs() < 2.01).sum()
                  tot_three = (((output/self.v_threshold - self.snn_rate_opt)*self.presim_len).abs() < 3.01).sum()

                  err_zero = tot_zero/tot_num
                  err_one = (tot_one-tot_zero)/tot_num
                  err_two = (tot_two-tot_one)/tot_num
                  err_three = (tot_three-tot_two)/tot_num
                  
                  self.arr2.append(err_zero)
                  self.arr2.append(err_one)
                  self.arr2.append(err_two)
                  self.arr2.append(err_three)
                               
                  return output
       
                 
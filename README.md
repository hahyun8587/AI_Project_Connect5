## Policy Gradient  
- 강화학습의 정책을 경사상승법으로 구하는 방법  
- 목적함수: 보상함수(모든 상태에서 얻을 수 있는 보상의 합)
- 보상함수를 증가시키는 방향으로 가중치 업데이트 -> 경사상승

## Neural Net Architecture  
- 입력층 노드 수: 226개(15x15) + bias  
- 출력층 노드 수: 225개(15x15) (각 액션의 확률)  
- 활성화 함수: relu  
- 출력층 활성화 함수: relu + softmax  

## Objective Function    
$\quad J(\theta) = \sum_{t=0}^{T-1} \log\pi_\theta (a_t|s_t)G_t$   
  
$\quad\pi_\theta(a_t|s_t)$: 상태 $s_t$에서 액션 $a_t$를 할 확률  
$\quad G_t$: 모든 보상의 합($\mathbb{E}[\gamma^0r_1+\gamma^1r_2+...+\gamma^{T-1}r_T|\pi_\theta]$)  
  
### gradient  
$\quad\nabla_{w_{jk}} J(\theta) = \sum_{t=0}^{T-1}\nabla_{w_{jk}}\log\pi_w (a_t|s_t)G_t$  
  
$\qquad\qquad\;= \sum_{t=0}^{T-1}(y_k^{(t)}-softmax(a_k^{(t)}))relu^\prime(z_k^{(t)})a_j $
  
$\qquad\qquad\;= \sum_{t=0}^{T-1}(y_k^{(t)}-\hat y_k^{(t)})relu^\prime(z_k^{(t)})a_j $  

경사상승법이므로 역전파 단계에서 미분값을 더해주면 된다.    
$w_{jk} \leftarrow \alpha (y_k^{(t)}-\hat y_k^{(t)})relu^\prime(z_k^{(t)})a_j$

## Algorithm

__for__ each timestep __do__  
$\quad$ feed forward  
$\quad$ save A and Z  
$\quad$ get reward from environment  
$\quad$ save reward  
__end for__  

rearrange A and Z

__for__ each episode __do__  
$\quad$ __for__ i = 0 to i = n - 1 __do__  
$\qquad W[i] \leftarrow  \delta W[i]$  
$\quad$ __end for__  
__end for__  

1. 신경망으로 액션 출력  
2. 환경으로부터 보상 저장  
3. 역전파    
4. 1~3 반복
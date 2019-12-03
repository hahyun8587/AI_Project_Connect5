html header: <script type="text/javascript"  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# AI_Project_Connect5

### Policy Gradient  
    * 강화학습의 정책을 경사상승법으로 구하는 방법  
    * 목적함수: 보상함수(모든 상태에서 얻을 수 있는 보상의 합)
    * 보상함수를 증가시키는 방향으로 가중치 업데이트 -> 경사상승

### Object Function
    $\nalba\theta$  

    J(w_jk) = \sum_s d_pi(s) \sum_a pi_w_jk(a\right\vert s)q_pi(s,a)

    $d_pi(s)$: 상태 s에 존재할 확률
    $pi_w_jk(a\right\vert s)$:상태 s에서 액션 a를 할 확률
    q_pi(s,a): 상태 s에서 액션 a를 해서 얻을 수 있는 모든 보상의 합

    => 


    $\gradient J(w_jk) = \sum_s d_pi(s) \sum_a \gradient_w_jk pi_w_jk(a\right\vert s)q_pi(s,a)
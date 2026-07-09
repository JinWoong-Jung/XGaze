# Results

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">GazeFollow</th>
      <th colspan="3" align="center">VideoAttentionTarget</th>
      <th colspan="3" align="center">ChildPlay</th>
    </tr>
    <tr>
      <th align="center">AUC &uarr;</th>
      <th align="center">Avg L2 &darr;</th>
      <th align="center">Min L2 &darr;</th>
      <th align="center">AUC &uarr;</th>
      <th align="center">L2 &darr;</th>
      <th align="center">AP<sub>in/out</sub> &uarr;</th>
      <th align="center">AUC &uarr;</th>
      <th align="center">L2 &darr;</th>
      <th align="center">AP &uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Gaze-LLE (ViT-B)</td>
      <td align="center">0.956</td><td align="center">0.104</td><td align="center">0.045</td>
      <td align="center">0.933</td><td align="center">0.107</td><td align="center">0.897</td>
      <td align="center">0.949</td><td align="center">0.106</td><td align="center">0.994</td>
    </tr>
    <tr>
      <td align="left">Gaze-LLE (ViT-L)</td>
      <td align="center">0.958</td><td align="center">0.099</td><td align="center">0.041</td>
      <td align="center">0.937</td><td align="center">0.103</td><td align="center">0.903</td>
      <td align="center">0.951</td><td align="center">0.101</td><td align="center">0.994</td>
    </tr>
    <tr>
      <td align="left"><b>Ours*</b></td>
      <td align="center">0.950</td><td align="center">0.098</td><td align="center">0.044</td>
      <td align="center">0.969</td><td align="center">0.095</td><td align="center">0.658</td>
      <td align="center">0.975</td><td align="center">0.093</td><td align="center">0.972</td>
    </tr>
  </tbody>
</table>

\* **Ours** — VideoAttentionTarget and ChildPlay results are **zero-shot** (no fine-tuning); GazeFollow is the trained result.

**Notes**
- **GazeFollow (Ours)**: trained-from-scratch test metrics (`logs/gazefollow/xgaze-gf_3525.out`). `Avg L2` = distance to the averaged GT point (`dist_to_avg`), `Min L2` = `min_dist`.
- **VideoAttentionTarget / ChildPlay (Ours)**: zero-shot metrics (`logs/vat/zero-shot/xgaze-vat_3539.out`, `logs/childplay/zero-shot/xgaze-cp_3542.out`). Fine-tuning logs are not available yet (`logs/*/fine-tuning/` are empty).

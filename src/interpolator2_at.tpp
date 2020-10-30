template<class T>
void Interpolator<T>::interpolate_at_mix(
  const std::vector<std::vector<ind_t>>& perms,
  const std::vector<ind_t>& indices,
  const std::vector<double>& weights,
  bArray<T>& out,
  const ind_t to,
  const bool arbitrary_phase_allowed
) const {
  if (indices.size()==0 || weights.size()==0)
    throw std::logic_error("Interpolation requires input data!");
  ind_t b_{this->branches()}, s_{this->branch_span()};
  verbose_update("Combining\n",data_.extract(indices).to_string(),"with weights ", weights);
  T *ox = out.ptr(to);
  if (arbitrary_phase_allowed){
    const T *d0 = data_.ptr(indices[0]); // a reference point for the phase calculation
    for (size_t x=0; x<indices.size(); ++x){
      const T *dx = data_.ptr(indices[x]);
      for (ind_t b=0; b < b_; ++b){
        auto p = perms[x][b];
        T eith = brille::utils::antiphase(s_, d0+b*s_, dx+p*s_);
        for (ind_t s=0; s<s_; ++s) ox[b*s_+s] += weights[x]*eith*dx[p*s_+s];
      }
    }
  } else {
    for (size_t x=0; x<indices.size(); ++x){
      const T *dx = data_.ptr(indices[x]);
      for (ind_t b=0; b < b_; ++b) for (ind_t s=0; s<s_; ++s)
        ox[b*s_+s] += weights[x]*dx[perms[x][b]*s_+s];
    }
  }
}


template<class T>
void Interpolator<T>::interpolate_at_mix(
  const std::vector<std::vector<ind_t>>& perms,
  const std::vector<std::pair<ind_t,double>>& idx_wgt,
  bArray<T>& out,
  const ind_t to,
  const bool arbitrary_phase_allowed
) const {
  if (idx_wgt.size()==0)
    throw std::logic_error("Interpolation requires input data!");
  std::vector<int> dummy;
  ind_t b_{this->branches()}, s_{this->branch_span()};
  T * ox = out.ptr(to);
  if (arbitrary_phase_allowed){
    const T *d0 = data_.ptr(idx_wgt[0].first);
    std::transform(perms.begin(), perms.end(), idx_wgt.begin(), std::back_inserter(dummy),
    [&](const std::vector<ind_t>& perm, const std::pair<ind_t,double>& iw){
      const T *dx = data_.ptr(iw.first);
      for (ind_t b=0; b<b_; ++b)
      {
        auto p = perm[b];
        T eith = brille::utils::antiphase(s_, d0+b*s_, dx+p*s_);
        for (ind_t s=0; s<s_; ++s) ox[b*s_+s] += iw.second*eith*dx[p*s_+s];
      }
      return 1;
    });
  } else {
    std::transform(perms.begin(), perms.end(), idx_wgt.begin(), std::back_inserter(dummy),
    [&](const std::vector<ind_t>& perm, const std::pair<ind_t,double>& iw){
      const T *dx = data_.ptr(iw.first);
      for (ind_t b=0; b<b_; ++b)
      {
        for (ind_t s=0; s<s_; ++s) ox[b*s_+s] += iw.second*dx[perm[b]*s_+s];
      }
      return 1;
    });
  }
}

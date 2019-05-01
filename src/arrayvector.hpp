

template<typename T> T* ArrayVector<T>::datapointer(size_t i, size_t j) const {
	T *ptr = nullptr;
	if (i<this->size() && j<this->numel())
		ptr = this->data + (i*this->numel() + j);
	if (!ptr){
		printf("ArrayVector<T>::datapointer(i=%u,j=%u) but size()=%u, numel()=%u\n",i,j,this->size(),this->numel());
		throw std::domain_error("attempting to access out of bounds pointer");
	}
	return ptr;
}

template<typename T> T ArrayVector<T>::getvalue(const size_t i, const size_t j) const {
	T *ptr, out;
	ptr = this->datapointer(i,j);
	out = *ptr;
	return out;
}
template<typename T> ArrayVector<T> ArrayVector<T>::extract(const size_t i) const {
	if (i<this->size()){
		ArrayVector<T> out(this->numel(),1u,this->datapointer(i));
		return out;
	}
	throw std::out_of_range("The requested element of the ArrayVector does not exist");
}
template<typename T> ArrayVector<T> ArrayVector<T>::extract(const size_t n, const size_t *i) const {
	bool allinbounds = true;
	ArrayVector<T> out(this->numel(),0u);
	for (size_t j=0; j<n; j++) if ( !(i[j]<this->size()) ){ allinbounds=false; break; }
	if (allinbounds){
		out.resize(n);
		for (size_t j=0; j<n; j++) out.set(j, this->datapointer(i[j]) );
	}
	return out;
}
template<typename T> ArrayVector<T> ArrayVector<T>::extract(const ArrayVector<size_t>& idx) const{
	bool allinbounds = true;
	ArrayVector<T> out(this->numel(),0u);
	if (idx.numel() != 1u) throw std::runtime_error("copying an ArrayVector by index requires ArrayVector<size_t> with numel()==1 [i.e., an ArrayScalar]");
	for (size_t j=0; j<idx.size(); ++j) if (idx.getvalue(j)>=this->size()){allinbounds=false; break;}
	if (allinbounds){
		out.resize(idx.size());
		for (size_t j=0; j<idx.size(); ++j) out.set(j, this->datapointer( idx.getvalue(j)) );
	}
	return out;
}
template<typename T> bool ArrayVector<T>::get(const size_t i, T* out) const {
	if (i>this->size()-1) return false;
	for (size_t j=0; j<this->numel(); j++) out[j]= this->getvalue(i,j);
	return true;
}
template<typename T> bool ArrayVector<T>::set(const size_t i, const T* in){
	if (i>this->size()-1) return false;
	for (size_t j=0; j<this->numel(); j++) this->data[i*this->numel()+j] = in[j];
	return true;
}
template<typename T> bool ArrayVector<T>::set(const size_t i, const ArrayVector<T>* in){
	if (i>this->size()-1 || this->numel()!=in->numel() || in->size()<1u ) return false;
	for (size_t j=0; j<this->numel(); j++) this->insert( in->getvalue(0,j), i,j) ;
	return true;
}
template<typename T> bool ArrayVector<T>::set(const size_t i, const ArrayVector<T>& in){
	if (i>this->size()-1 || this->numel()!=in.numel() || in.size()<1u ) return false;
	for (size_t j=0; j<this->numel(); j++) this->insert( in.getvalue(0,j), i,j) ;
	return true;
}
template<typename T> bool ArrayVector<T>::insert(const T in, const size_t i, const size_t j){
	bool inrange = i<this->size() && j<this->numel();
	if (inrange) this->data[i*this->numel()+j] = in;
	return inrange;
}
template<typename T> void ArrayVector<T>::printformatted(const char * fmt,const size_t first, const size_t last, const char * after) const {
	size_t i,j,b=this->numel();
	for (i=first;i<last;i++){ for (j=0;j<b;j++) printf(fmt,this->getvalue(i,j)); printf(after);	}
}
template<typename T> void ArrayVector<T>::print() const {
	const char * fmt = std::is_floating_point<T>::value ? " % g " : " % d ";
	this->printformatted(fmt,0,this->size(),"\n");
}
template<typename T> void ArrayVector<T>::print(const size_t i) const {
	this->print(i,i,"\0");
}
template<typename T> void ArrayVector<T>::print(const size_t first, const size_t last, const char *after) const {
	const char * fmt = std::is_floating_point<T>::value ? " % g " : " % d ";
	if (first<this->size() && last<this->size())
		this->printformatted(fmt,first,last+1,after);
	else
		printf("Attempted to print elements %u to %u of size()=%u ArrayVector!\n",first,last,this->size());
}

template<typename T> void ArrayVector<T>::printheader(const char* name) const {
	printf("%s numel %u, size %u\n", name, this->numel(), this->size());
}

template<typename T> size_t ArrayVector<T>::resize(size_t newsize){
	bool std = (newsize*this->numel())>0;
	T * newdata;
	// allocate a new block of memory
	if (std) newdata = safealloc<T>(newsize*this->numel());
	if (this->size()*this->numel()) { // copy-over data :(
		size_t smallerN = (this->size() < newsize) ? this->size() : newsize;
		for (size_t i=0; i<smallerN*this->numel(); i++) newdata[i] = this->data[i];
		// hand-back the chunk of memory which data points to
		delete[] this->data;
	}
	// and set data to the newdata pointer;
	this->N = newsize;
	if (std) this->data = newdata;
	return newsize;
}
template<typename T> size_t ArrayVector<T>::refresh(size_t newnumel, size_t newsize){
	// first off, remove the old data block, if it exists
	if (this->size()*this->numel())	delete[] this->data;
	bool std = (newsize*newnumel)>0;
	T * newdata;
	// allocate a new block of memory
	if (std) newdata = safealloc<T>(newsize*newnumel);
	// and set data to the newdata pointer;
	this->M = newnumel;
	this->N = newsize;
	this->data = std ? newdata : nullptr;
	return newnumel*newsize;
}






template<typename T> template<typename R> bool ArrayVector<T>::isapprox(const ArrayVector<R> &that) const {
	AVSizeInfo si = this->consistency_check(that);
	if (si.scalara^si.scalarb) return false; // only one is an "ArrayScalar"
	for (size_t i=0; i<si.n; i++)
		for (size_t j=0; j<si.m; j++)
			if ( !approx_scalar(this->getvalue(si.oneveca?0:i,j), that.getvalue(si.onevecb?0:i,si.singular?0:j)) )
				return false;
	return true;
}
template<typename T> bool ArrayVector<T>::isapprox(const size_t i, const size_t j) const {
	for (size_t k=0; k<this->numel(); k++) if (!approx_scalar(this->getvalue(i,k),this->getvalue(j,k))) return false;
	return true;
}

template<typename T> void ArrayVector<T>::cross(const size_t i, const size_t j, T* out) const {
	if (this->numel()!=3u) throw std::domain_error("cross is only defined for 3-D vectors");
	if (i<this->size()&&j<this->size())	vector_cross(out,this->datapointer(i,0),this->datapointer(j,0));
	else throw std::domain_error("attempted to access out of bounds memory");
}
template<typename T> T ArrayVector<T>::dot(const size_t i, const size_t j) const {
	T out = 0;
	for (size_t k=0; k<this->numel(); k++) out += this->getvalue(i,k)*this->getvalue(j,k);
	return out;
}
template<typename T> T ArrayVector<T>::norm(const size_t i) const {
	return sqrt(this->dot(i,i));
}


template<typename T> bool ArrayVector<T>::arealltrue(void) const {
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			if (!this->getvalue(i,j)) return false;
	return true;
}
template<typename T> bool ArrayVector<T>::areanytrue(void) const {
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			if (this->getvalue(i,j)) return true;
	return false;
}
template<typename T> bool ArrayVector<T>::areallpositive(void) const {
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			if (this->getvalue(i,j)<0) return false;
	return true;
}
template<typename T> bool ArrayVector<T>::areallzero(void) const {
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			if (this->getvalue(i,j)) return false;
	return true;
}

template<typename T> ArrayVector<int> ArrayVector<T>::round() const{
	ArrayVector<int> out(this->numel(),this->size());
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			out.insert( std::round(this->getvalue(i,j)), i,j);
	return out;
}
template<typename T> ArrayVector<int> ArrayVector<T>::floor() const{
	ArrayVector<int> out(this->numel(),this->size());
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			out.insert( std::floor(this->getvalue(i,j)), i,j);
	return out;
}
template<typename T> ArrayVector<int> ArrayVector<T>::ceil() const{
	ArrayVector<int> out(this->numel(),this->size());
	for (size_t i=0; i<this->size(); i++)
		for (size_t j=0; j<this->numel(); j++)
			out.insert( std::ceil(this->getvalue(i,j)), i,j);
	return out;
}

template<typename T> ArrayVector<T> ArrayVector<T>::sum( const int dim ) const {
	T tmp;
	ArrayVector<T> out;
	switch (dim){
		case 1:
			out.refresh(1u,this->size());
			for (size_t i=0; i<this->size(); i++){
				tmp = T(0);
				for (size_t j=0; j<this->numel(); j++) tmp += this->getvalue(i,j);
				out.insert(tmp, i,0);
			}
			break;
		default:
			out.refresh(this->numel(),1u);
			for (size_t j=0; j<this->numel(); j++){
				tmp = T(0);
				for (size_t i=0; i<this->size(); i++) tmp += this->getvalue(i,j);
				out.insert(tmp, 0,j);
			}
			break;
	}
	return out;
}

template<class T, class R, template<class> class A,
         typename=typename std::enable_if< std::is_base_of<ArrayVector<T>,A<T>>::value && !std::is_base_of<LatVec,A<T>>::value>::type,
				 class S = typename std::common_type<T,R>::type
				 >
A<S> dot(const A<T>& a, const A<R>& b){
	AVSizeInfo si = a.consistency_check(b);
	if (si.scalara^si.scalarb) throw std::runtime_error("ArrayVector dot requires equal numel()");
	A<S> out(1u,si.n);
	S tmp;
	for (size_t i=0; i<si.n; ++i){
		tmp = S(0);
		for (size_t j=0; j<si.m; ++j) tmp+= a.getvalue(si.oneveca?0:i,j) * b.getvalue(si.onevecb?0:i,j);
		out.insert(tmp,i,0);
	}
	return out;
}


template<class T, template<class> class L,typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,L<T>>::value>::type>
L<int> round(const L<T>& a){
	L<int> out(a);
	for (size_t i=0; i<a.size(); i++) for (size_t j=0; j<a.numel(); j++) out.insert( std::round(a.getvalue(i,j)), i,j);
	return out;
}
// floor(LatVec)
template<class T, template<class> class L,typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,L<T>>::value>::type>
L<int> floor(const L<T>& a){
	L<int> out(a);
	for (size_t i=0; i<a.size(); i++) for (size_t j=0; j<a.numel(); j++) out.insert( std::floor(a.getvalue(i,j)), i,j);
	return out;
}
// ceil(LatVec)
template<class T, template<class> class L,typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,L<T>>::value>::type>
L<int> ceil(const L<T>& a){
	L<int> out(a);
	for (size_t i=0; i<a.size(); i++) for (size_t j=0; j<a.numel(); j++) out.insert( std::ceil(a.getvalue(i,j)), i,j);
	return out;
}



// In Place arithmetic ArrayVector +-*/ ArrayVector
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator +=(const ArrayVector<T> &av){
	AVSizeInfo si = this->inplace_consistency_check(av);
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) this->insert( this->getvalue(i,j) + av.getvalue(si.onevecb?0:i,si.singular?0:j), i,j );
	return *this;
}
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator -=(const ArrayVector<T> &av){
	AVSizeInfo si = this->inplace_consistency_check(av);
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) this->insert( this->getvalue(i,j) - av.getvalue(si.onevecb?0:i,si.singular?0:j), i,j );
	return *this;
}
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator *=(const ArrayVector<T> &av){
	AVSizeInfo si = this->inplace_consistency_check(av);
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) this->insert( this->getvalue(i,j) * av.getvalue(si.onevecb?0:i,si.singular?0:j), i,j );
	return *this;
}
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator /=(const ArrayVector<T> &av){
	AVSizeInfo si = this->inplace_consistency_check(av);
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) this->insert( this->getvalue(i,j) / av.getvalue(si.onevecb?0:i,si.singular?0:j), i,j );
	return *this;
}
// In-place binary operators with scalars
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator +=(const T& av){
	for (size_t i=0; i<this->size(); i++) for(size_t j=0; j<this->numel(); j++) this->insert( this->getvalue(i,j) + av, i,j );
	return *this;
}
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator -=(const T& av){
	for (size_t i=0; i<this->size(); i++) for(size_t j=0; j<this->numel(); j++) this->insert( this->getvalue(i,j) - av, i,j );
	return *this;
}
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator *=(const T& av){
	for (size_t i=0; i<this->size(); i++) for(size_t j=0; j<this->numel(); j++) this->insert( this->getvalue(i,j) * av, i,j );
	return *this;
}
template<typename T> ArrayVector<T>& ArrayVector<T>:: operator /=(const T& av){
	for (size_t i=0; i<this->size(); i++) for(size_t j=0; j<this->numel(); j++) this->insert( this->getvalue(i,j) / av, i,j );
	return *this;
}


template<class T, class R, template<class> class A,
         typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator+(const A<T>& a, const A<R>& b){
	AVSizeInfo si = a.consistency_check(b);
	A<S> out( si.aorb ? a : b);
	out.refresh(si.m,si.n); // in case a.size == b.size but one is singular, or a.numel == b.numel but one is scalar
	// if (si.oneveca || si.onevecb || si.scalara || si.scalarb){
	// 	printf("=======================\n            %3s %3s %3s\n","A","B","A+B");
	// 	printf("OneVector   %3d %3d\n",si.oneveca?1:0,si.onevecb?1:0);
	// 	printf("ArrayScalar %3d %3d\n",si.scalara?1:0,si.scalarb?1:0);
	// 	printf("-----------------------\n");
	// 	printf("chosen      %3d %3d\n",si.aorb?1:0,si.aorb?0:1);
	// 	printf("size()      %3u %3u %3u\n",a.size(), b.size(), out.size());
	// 	printf("numel()     %3u %3u %3u\n",a.numel(), b.numel(), out.numel());
	// }
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) out.insert( a.getvalue(si.oneveca?0:i,si.scalara?0:j) + b.getvalue(si.onevecb?0:i,si.scalarb?0:j), i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator-(const A<T>& a, const A<R>& b){
	AVSizeInfo si = a.consistency_check(b);
	A<S> out( si.aorb ? a : b);
	out.refresh(si.m,si.n); // in case a.size == b.size but one is singular, or a.numel == b.numel but one is scalar
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) out.insert( a.getvalue(si.oneveca?0:i,si.scalara?0:j) - b.getvalue(si.onevecb?0:i,si.scalarb?0:j), i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator*(const A<T>& a, const A<R>& b){
	AVSizeInfo si = a.consistency_check(b);
	A<S> out( si.aorb ? a : b);
	out.refresh(si.m,si.n); // in case a.size == b.size but one is singular, or a.numel == b.numel but one is scalar
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) out.insert( a.getvalue(si.oneveca?0:i,si.scalara?0:j) * b.getvalue(si.onevecb?0:i,si.scalarb?0:j), i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 class S = typename std::common_type<T,R>::type,
				 typename=typename std::enable_if<std::is_floating_point<S>::value>::type >
A<S> operator/(const A<T>& a, const A<R>& b){
	AVSizeInfo si = a.consistency_check(b);
	A<S> out( si.aorb ? a : b);
	out.refresh(si.m,si.n); // in case a.size == b.size but one is singular, or a.numel == b.numel but one is scalar
	for (size_t i=0; i<si.n; i++) for(size_t j=0; j<si.m; j++) out.insert( a.getvalue(si.oneveca?0:i,si.scalara?0:j) / b.getvalue(si.onevecb?0:i,si.scalarb?0:j), i,j );
	return out;
}

template<class T, class R, template<class> class A,
         typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator+(const A<T>& a, const R& b){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( a.getvalue(i,j) + b, i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator-(const A<T>& a, const R& b){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( a.getvalue(i,j) - b, i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator*(const A<T>& a, const R& b){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( a.getvalue(i,j) * b, i,j );
	return out;
}
// template<class T, class R, template<class> class A,
// 				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
// 				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
// 				 class S = typename std::common_type<T,R>::type,
// 				 typename=typename std::enable_if<std::is_floating_point<S>::value>::type >
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type> // leave off the is_floating_point restriction on S for the special case used by halfN
A<S> operator/(const A<T>& a, const R& b){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( a.getvalue(i,j) / b, i,j );
	return out;
}
template<class T, class R, template<class> class A,
         typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator+(const R& b, const A<T>& a){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( b + a.getvalue(i,j), i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator-(const R& b, const A<T>& a){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( b - a.getvalue(i,j), i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type >
A<S> operator*(const R& b, const A<T>& a){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( b * a.getvalue(i,j), i,j );
	return out;
}
template<class T, class R, template<class> class A,
				 typename=typename std::enable_if<std::is_base_of<ArrayVector<T>,A<T>>::value>::type,
				 typename=typename std::enable_if<!std::is_base_of<ArrayVector<R>,R>::value>::type,
				 class S = typename std::common_type<T,R>::type,
				 typename=typename std::enable_if<std::is_floating_point<S>::value>::type >
A<S> operator/(const R& b, const A<T>& a){
	A<S> out(a);
	for (size_t i=0; i<out.size(); i++) for(size_t j=0; j<out.numel(); j++) out.insert( b / a.getvalue(i,j), i,j );
	return out;
}


template<typename T> ArrayVector<T> ArrayVector<T>:: operator -() const {
	ArrayVector<T> out(this->numel(),this->size());
	for (size_t i=0; i<(this->numel()*this->size()); i++) out.data[i] = -(this->data[i]);
	return out;
}

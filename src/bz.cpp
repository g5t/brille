#include <iostream>
#include <cstring>
#include <limits>

#include "bz.h"
#include "arrayvector.h"
#include "lattice.h"
#include "latvec.h"


BrillouinZone::BrillouinZone(Reciprocal lat,int extent): lattice(lat) {this->determine_everything(extent); }
void BrillouinZone::set_vertices(ArrayVector<double> newverts){
	if (newverts.numel()!=3u) throw "BrillouinZone objects only take 3-D vectors for vertices";
	this->vertices = newverts;
}
void BrillouinZone::set_faces(ArrayVector<int> newfaces){
	if (newfaces.numel()!=3u) throw "BrillouinZone objects only take 3-D vectors for faces";
	this->faces = newfaces;
}
void BrillouinZone::set_faces_per_vertex(ArrayVector<int> newfpv){
	if (newfpv.numel()!=3u) throw "BrillouinZone vertices are the intersections of three faces each";
	this->faces_per_vertex = newfpv;
}

void BrillouinZone::determine_everything(const int extent){
	// LQVec<int> tau(this->lattice);
	// int ntau = make_all_indices(&tau,extent);
	LQVec<int> tau(this->lattice, make_relative_neighbour_indices(extent) );
	int ntau = (int)tau.size();
	// the number of unique combinations of 3-taus is the number that we need to check
	size_t ntocheck=0;
	// there is probably a better way to do this, but brute force never hurt anyone
	for (int i=0; i<(ntau-2); i++) for (int j=i+1; j<(ntau-1); j++) for (int k=j+1; k<ntau; k++) ntocheck++;

	LQVec<double> all_vertices(this->lattice,ntocheck);
	ArrayVector<int> all_ijk(3,ntocheck);

	// LQVec<double> tauhat(this->lattice), halftau(this->lattice);
	// ArrayVector<double> lentau = tau.norm();
	ArrayVector<double> lentau = norm(tau);
	LQVec<double> tauhat = tau/lentau;
	double two = 2;
	LQVec<double> halftau(tau/two);

	ArrayVector<double> tauhat_xyz;
	tauhat_xyz = tauhat.get_xyz();

	int count=0;
	for (int i=0; i<(ntau-2); i++){
		for (int j=i+1; j<(ntau-1); j++){
			for (int k=j+1; k< ntau   ; k++){
				if ( three_plane_intersection(&tauhat, &halftau, &tauhat_xyz, i,j,k, &all_vertices, count) ){
					all_ijk.insert(i, count, 0); //insert value i at position (count,0)
					all_ijk.insert(j, count, 1);
					all_ijk.insert(k, count, 2);
					count++;
				}
			}
		}
	}
	// there are count intersections of three planes (strictly count<=ntocheck, but probably count < ntocheck/2)

	// next we need to check for the kernel of intersection points which are closer to the origin than any (non-intersection-defining) planes
	LQVec<double> in_verts(this->lattice,count);
	ArrayVector<int> in_ijk(3,count);
	int in_cnt = 0;
	for (int i=0; i<count; i++){
		// this between_origin_and_plane expects all vectors in an orthonormal frame
		if ( between_origin_and_plane( &halftau, &all_vertices, &all_ijk, i, &in_verts, in_cnt, 1e-10 ) ){
			in_ijk.set(in_cnt++, all_ijk.datapointer(i));
		}
	}
	// if ( in_cnt > 0) in_verts.print(0,in_cnt-1);

	// it's possible that multiple three-plane intersections have given the same
	// intersection point -- e.g., for a cubic system the intersection points of
	// (100),(010),(001); (110),(010),(001); (100),(110),(001); (100),(010),(011)
	// (101),(010),(001); ... are all the same point, (111).
	// The true vertex of the Brillouin Zone is the intersection of the three
	// planes with smallest norm

	// First, find the first vertex which is unique of equivalent vertices
	bool *vertisunique = new bool[in_cnt]();
	for (int i=0; i<in_cnt; i++) vertisunique[i] = true;
	for (int i=0; i<in_cnt-1; i++){
		if (vertisunique[i]){
			for (int j=i+1;j<in_cnt; j++){
				if (vertisunique[j] && in_verts.isapprox(i,j)){
					// printf("vert %d == %d\n",i,j);
					 vertisunique[j] = false;
				 }
			}
		}
	}
	// count up the unique vertices and keep track of their indices
	int unqcnt=0, *unqidx = new int[in_cnt]();
	for (int i=0; i<in_cnt; i++) if (vertisunique[i]) unqidx[unqcnt++]=i;

	//printf("%d unique vertices\n",unqcnt);

	// create a mapping which holds the indices of all equivalent vertices
	// so unqidxmap[1,:] are all of the indices which equal the first unique vertex
	int *unqidxmap = new int[in_cnt*unqcnt]();
	int *numidxmap = new int[unqcnt]();
	for (int i=0; i<unqcnt; i++){
		numidxmap[i]=1;
		unqidxmap[i*in_cnt] = unqidx[i];
		for (int j=0; j<in_cnt; j++){
			if (unqidx[i]!=j && !vertisunique[j] && in_verts.isapprox(unqidx[i],j))
				unqidxmap[ i*in_cnt + numidxmap[i]++ ] = j;
		}
	}
	delete[] vertisunique;
	// and determine the "length" of the planes which define each vertex, maintaining
	// the equivalence relationship already established, but allocating only the
	// memory actually needed by first finding the maximum number intersections
	// which gave the same vertex
	int maxequiv = 0;
	for (int i=0; i<unqcnt; i++) if ( numidxmap[i]>maxequiv) maxequiv=numidxmap[i];
	double *unqlenmap = new double[maxequiv*unqcnt](); // no need to allocate in_cnt*unqcnt memory when maxequiv is known
	for (int i=0; i<unqcnt; i++){
		for (int j=0; j<numidxmap[i]; j++){
			unqlenmap[i*maxequiv + j] = 0;
			for (int k=0; k<3; k++){
				unqlenmap[i*maxequiv + j] += halftau.norm( in_ijk.getvalue(unqidxmap[i*in_cnt+j],k));
			}
		}
	}
	// use the "length" information to select which equivalent vertex we should keep
	int *minequividx = new int[unqcnt]();
	double *minequivlen = new double[unqcnt]();
	for (int i=0; i<unqcnt; i++){
		minequivlen[i] = std::numeric_limits<double>::max(); // better than 1./0.
		for (int j=0; j<numidxmap[i]; j++){
			if ( unqlenmap[i*maxequiv +j] < minequivlen[i]){
				minequividx[i] = unqidxmap[i*in_cnt+j];
				minequivlen[i] = unqlenmap[i*maxequiv +j];
			}
		}
	}
	delete[] unqidx;
	delete[] numidxmap;
	delete[] unqidxmap;
	delete[] unqlenmap;
	delete[] minequivlen;

	if (unqcnt == 0) 	throw std::runtime_error("No unique vertices found?!");

	LQVec<double> unq_vrt(this->lattice, unqcnt);
	ArrayVector<int> unq_ijk(3,unqcnt);
	for (int i=0; i<unqcnt; i++){
			unq_vrt.set( i, in_verts.datapointer(minequividx[i]) );
			unq_ijk.set( i, in_ijk.datapointer(minequividx[i]) );
	}
	delete[] minequividx;

	// store the reciprocal space positions of the vertices of the first Brillouin Zone
	this->set_vertices(unq_vrt); // does this work with LQVec smlst_vrt?

	// determine which of the taus actually contribute to at least one vertex
	int ncontrib=0, *contrib = new int[ntau]();
	for (int i=0; i<ntau; i++)
	for (int j=0; j<unqcnt; j++)
	if ( unq_ijk.getvalue(j,0) == i || unq_ijk.getvalue(j,1) == i || unq_ijk.getvalue(j,2) == i ){
		contrib[ncontrib++]=i;
		break;
	}
	ArrayVector<int> faces(3,ncontrib);
	for (int i=0; i<ncontrib; i++) faces.set(i, tau.datapointer(contrib[i]));
	this->set_faces(faces);

	// Each vertex is the intersection of three faces -- smlst_ijk contains the indexes into tau
	// but since tau contains planes which do not contribute to the first Brillouin Zone
	// we still have work to do. Replace the indices in smlst_ijk with their equivalent
	// indices into this->faces (using contrib as the map)
	ArrayVector<int> fpv(3,unqcnt);
	for (int i=0; i<unqcnt; i++)
	for (int j=0; j<3; j++)
	for (int k=0; k<ncontrib; k++){
		if ( unq_ijk.getvalue(i,j) == contrib[k] ){
			fpv.insert(k,i,j);
			break;
		}
	}
	this->set_faces_per_vertex(fpv);

	delete[] contrib;
}

size_t BrillouinZone::get_vertices_bare(const size_t max, double *out) const {
	if (max<this->vertices.size()) return 0;
	for (size_t i=0; i<this->vertices.size(); i++)
		this->vertices.get(i, out+i*3);
	return this->vertices.size();
}
size_t BrillouinZone::get_faces_bare(const size_t max, int *out) const {
	if (max<this->faces.size()) return 0;
	for (size_t i=0; i<this->faces.size(); i++)
		this->faces.get(i, out+i*3);
  return this->faces.size();
}
size_t BrillouinZone::get_faces_per_vertex_bare(const size_t max, int *out) const {
	if (max<this->faces_per_vertex.size()) return 0;
	for (size_t i=0; i<this->faces_per_vertex.size(); i++)
		this->faces_per_vertex.get(i, out+i*3);
	return this->faces_per_vertex.size();
}
LQVec<double> BrillouinZone::get_vertices(void) const { return LQVec<double>(this->lattice, this->vertices); }
LQVec<int>    BrillouinZone::get_faces   (void) const { return LQVec<int>(this->lattice, this->faces   ); }
ArrayVector<int> BrillouinZone::get_faces_per_vertex(void) const {
	ArrayVector<int> out = this->faces_per_vertex; // make sure we return a copy, not the internal object
	return out;
}

void BrillouinZone::print() const {
	printf("BrillouinZone with %d vertices and %d faces\n",this->vertices_count(),this->faces_count());
}

template<typename T> ArrayVector<bool> BrillouinZone::isinside(const LQVec<T> *p, const double tol){
	// this BrillouinZone object has already been instantiated, meaning that it
	// knows *which* reciprocal lattice points define it!
	// we just need to check whether the point(s) in p are closer to the origin than the planes which make-up the Brillouin Zone
	ArrayVector<bool> out(1u,p->size());
	ArrayVector<double> inout(1u,p->size());
	LQVec<double> facevecs(this->lattice, (this->faces)/2.0); // this->faces is a) ArrayVector<integer> and b) reciprocal lattice points, and we want their halves
	bool tmp = true;
	LQVec<T> p_i;
	for (size_t i=0; i<p->size(); i++){
		// {p[i] - (hkl)}⋅(hkl) -- positive means p is beyond the plane defined by (hkl)
		// inout = (p->get(i)-facevecs).dot(&facevecs);
		p_i = p->get(i);
		if (p_i.size()!=1u) throw std::runtime_error("error accessing p element");
		inout = dot(facevecs, p_i-facevecs);
		tmp = true;
		for (int j=0; j<facevecs.size(); j++){
			if ( inout.getvalue(j) > tol ) {
				tmp = false;
				break;
			}
		}
		out.insert(tmp,i);
	}
	return out;
}
// template<typename T> ArrayVector<bool> BrillouinZone::isinside(const LQVec<T>& p, const double tol){
// 	ArrayVector<bool> out(1u,p.size());
// 	ArrayVector<double> inout(1u,p.size());
// 	LQVec<double> facevecs(this->lattice, (this->faces)/2.0); // this->faces is a) ArrayVector<integer> and b) reciprocal lattice points, and we want their halves
// 	bool tmp = true;
// 	for (size_t i=0; i<p.size(); i++){
// 		inout = dot( p[i]-facevecs, facevecs);
// 		tmp = true;
// 		for (int j=0; j<facevecs.size(); j++){
// 			if ( inout.getvalue(j) > tol ) {
// 				tmp = false;
// 				break;
// 			}
// 		}
// 		out.insert(tmp,i);
// 	}
// 	return out;
// }

bool BrillouinZone::moveinto(const LQVec<double> *Q, LQVec<double> *q, LQVec<int> *tau){
	// we could enforce that q and tau already have enough storage space to hold
	// their respective parts of Q, but LQVec/ArrayVector objects will resize
	// themselves if necessary, so don't bother.
	ArrayVector<bool> allinside = this->isinside(Q);

	// *tau = *Q * (int)(0); //hopefully this calls operator*<T,int> which *should* return an LQVec<int>
	// for (size_t i=0; i<tau->size(); ++i) for (size_t j=0; j<tau->numel(); ++j) tau->insert(0,i,j); // this sucks
	*q   = *Q * 0.0;
	*tau = round(*q); // returns int element type for sure

	LQVec<int> facehkl = this->get_faces();
	// ArrayVector<double> facelen = facehkl.norm();
	ArrayVector<double> facelen = norm(facehkl);
	LQVec<double> facenrm = facehkl/facelen;
	LQVec<double> qi;
	LQVec<int> taui;
	ArrayVector<int> Nhkl;
	int maxat = 0;
	int maxnm = 0;
	size_t count =0;
	for (size_t i=0; i<Q->size(); i++){
		count = 0;
		qi = Q->get(i);
		taui = tau->get(i);
		while (!allinside.getvalue(i) && count++ < 100*facelen.size()){
			Nhkl = (dot( qi , facenrm )/facelen).round();
			if ( Nhkl.areallzero() ) {allinside.insert(true,i); break;} // qi is *on* the Brilluoin Zone surface (or inside) so break.
			maxnm = 0;
			maxat = 0;
			for (size_t j=0; j<Nhkl.size(); ++j){
				if (Nhkl.getvalue(j)>maxnm){
					maxnm = Nhkl.getvalue(j);
					maxat = j;
				}
			}
			qi -= facehkl[maxat] * (double)(maxnm); // ensure we subtract LQVec<double>
			taui += facehkl[maxat] * maxnm; // but add LQVec<int>

			allinside.insert(this->isinside(&qi).getvalue(0), i);
		}
		q->set(i, &qi);
		tau->set(i, &taui);
	}
	return allinside.arealltrue(); // return false if any points are still outside of the first Brilluoin Zone
}

bool three_plane_intersection(const LQVec<double> *n,                // plane normals
	                            const LQVec<double> *p,                // a point on each plane
															const ArrayVector<double> *xyz,        // the plane normals in a inverse Angstrom orthonormal coordinate system
															const int i, const int j, const int k, // indices into n, p, xyz
															LQVec<double> *iat,                    // output storage array of intersection points
															const int idx)                         // the index where the intersection is inserted if found
															{
	// we need to check whether the matrix formed by the orthonormal-frame components of the three planes is nearly-singular
	double *M = new double[9];
	xyz->get(i, M);
	xyz->get(j, M+3);
	xyz->get(k, M+6);
	double detM;
	detM = matrix_determinant(M);
	delete[] M;
	if ( my_abs(detM) > 1e-10 ){
		LQVec<double> ni,nj,nk, pi,pj,pk, cij,cjk,cki, tmp;
		ni=n->get(i);	nj=n->get(j);	nk=n->get(k);
		pi=p->get(i);	pj=p->get(j);	pk=p->get(k);
		// cij=ni.cross(&nj);	cjk=nj.cross(&nk);	cki=nk.cross(&ni);
		cij=cross(ni,nj);	cjk=cross(nj,nk);	cki=cross(nk,ni);

		// tmp = cjk*(pi.dot(&ni)) + cki*(pj.dot(&nj)) + cij*(pk.dot(&nk));
		tmp = cjk*dot(pi,ni) + cki*dot(pj,nj) + cij*dot(pk,nk);
		tmp /= detM;

		iat->set(idx, tmp.datapointer() );
		return true;
	}
	return false;
}


bool between_origin_and_plane(const LQVec<double> *p,
	                            const LQVec<double> *v,
															const ArrayVector<int> *ijk,
															const int idx,
															LQVec<double> *inv,
															const int store_at,
															const double tol){
	if (ijk->numel()!=3u) throw "expected all three vector arrays";
	// p and v should be the points defining each plane and the vertices of the intersections of three planes
	ArrayVector<double> inout;
	// inout = (v->get(idx) - *p).dot(p);
	inout = dot(v->get(idx) - *p, *p);
	// we want to skip over the planes which gave us this intersection point
	size_t i, skip1, skip2, skip3;
	skip1 = (size_t) ijk->getvalue(idx,0);
	skip2 = (size_t) ijk->getvalue(idx,1);
	skip3 = (size_t) ijk->getvalue(idx,2);
	for (i=0; i < p->size(); i++)
		if ( !(i==skip1||i==skip2||i==skip3) && inout.getvalue(i) > tol)
			return false;
	// none of p are closer to the origin than v(i)
	inv->set(store_at, v->datapointer(idx));
	return true;
}

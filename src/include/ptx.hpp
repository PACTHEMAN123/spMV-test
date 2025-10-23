
#ifndef PTX_DEF_H
#define PTX_DEF_H

// global load inst.
// ld.global.<cache_modifier>.<type> dst, [addr];
// <cache_modifier> = ca: cache in L1 L2
// <cache_modifier> = cg: bypass L1, only cache in L2

#define LDG_F_CG_32(addr,res0) {asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(res0) : "l"(addr));}
#define LDG_F_CA_32(addr,res0) {asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(res0) : "l"(addr));}

#define LDG_U_CG_32(addr,res0) {asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(res0) : "l"(addr));}
#define LDG_U_CA_32(addr,res0) {asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(res0) : "l"(addr));}

#define LDG_F_CG_64(addr,res0,res1) {asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(res0), "=f"(res1) : "l"(addr));}
#define LDG_F_CA_64(addr,res0,res1) {asm volatile("ld.global.ca.v2.f32 {%0, %1}, [%2];" : "=f"(res0), "=f"(res1) : "l"(addr));}
#define LDG_F_CG_128(addr,res0,res1,res2,res3) {asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(res0), "=f"(res1), "=f"(res2), "=f"(res3) : "l"(addr));}
#define LDG_F_CA_128(addr,res0,res1,res2,res3) {asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(res0), "=f"(res1), "=f"(res2), "=f"(res3) : "l"(addr));}

// share memory async load
#define CP_ASYNC_CA_32(dst,src) {asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"::"l"(dst),"l"(src));}
#define CP_ASYNC_CG_32(dst,src) {asm volatile("cp.async.cg.shared.global [%0], [%1], 4;"::"l"(dst),"l"(src));}
#define CP_ASYNC_CA_32_PLEZ(dst,src,guard) {asm volatile("{\
	.reg .pred p_cpas;\r\n \
	.reg .b64 dst_s, src_g;\r\n \
	cvta.to.shared.u64 dst_s, %0;\r\n \
	cvta.to.global.u64 src_g, %1;\r\n \
	setp.ne.u32 p_cpas, %2,0;\r\n \
	@p_cpas cp.async.ca.shared.global [dst_s], [src_g], 4;\r\n \
	@!p_cpas st.shared.u32 [dst_s], 0;\r\n \
	}"::"l"(dst),"l"(src),"r"(guard));}
#define CP_ASYNC_COMMIT_GROUP() {asm volatile("cp.async.commit_group;");}
#define CP_ASYNC_WAIT_GROUP(N) {asm volatile("cp.async.wait_group %0;"::"n"(N));}
#define CP_ASYNC_WAIT_ALL() {asm volatile("cp.async.wait_all;");}

#endif

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

#endif
                                                
import triton
import triton.language as tl


@triton.jit
def fused_add_round_kernel(
    x_ptr,
    output_ptr,                                        
    seed,
    n_elements,
    EXPONENT_BIAS,
    MANTISSA_BITS,
    BLOCK_SIZE: tl.constexpr,
):
    ''
                                        
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

               
    x = tl.load(x_ptr + offsets, mask=mask)
    rand_vals = tl.rand(seed, offsets) - 0.5

    x = tl.cast(x, tl.float16)
    delta = tl.load(output_ptr + offsets, mask=mask)
    delta = tl.cast(delta, tl.float16)
    x = x + delta

    x_bits = tl.cast(x, tl.int16, bitcast=True)

                                                                                          
                                            
    fp16_exponent_bits = (x_bits & 0x7C00) >> 10
    fp16_normals = fp16_exponent_bits > 0
    fp16_exponent = tl.where(fp16_normals, fp16_exponent_bits - 15, -14)

                                                                                          
    exponent = fp16_exponent + EXPONENT_BIAS
    MAX_EXPONENT = 2 * EXPONENT_BIAS + 1
    exponent = tl.where(exponent > MAX_EXPONENT, MAX_EXPONENT, exponent)
    exponent = tl.where(exponent < 0, 0, exponent)

                                                               
                                                     
                                                       
                                                              
    eps_exp = tl.maximum(0, tl.minimum(31, exponent - EXPONENT_BIAS - MANTISSA_BITS + 15))

                                           
    eps_normal = tl.cast(tl.cast(eps_exp << 10, tl.int16), tl.float16, bitcast=True)

                                                             
                                                                    
                                        
    eps_subnormal = tl.cast(tl.cast((16 - EXPONENT_BIAS - MANTISSA_BITS) << 10, tl.int16), tl.float16, bitcast=True)
    eps = tl.where(exponent > 0, eps_normal, eps_subnormal)

                                
    eps = tl.where(x == 0, 0.0, eps)

                               
    output = tl.cast(x + rand_vals * eps, tl.bfloat16)

                      
    tl.store(output_ptr + offsets, output, mask=mask)

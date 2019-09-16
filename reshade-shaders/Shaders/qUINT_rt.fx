/*=============================================================================

	ReShade 4 effect file
    github.com/martymcmodding

	Support me:
   		paypal.me/mcflypg
   		patreon.com/mcflypg

    Path Traced Global Illumination 

    by Marty McFly / P.Gilcher
    part of qUINT shader library for ReShade 4

    CC BY-NC-ND 3.0 licensed.

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

//these two are required if I want to reuse the MXAO textures properly

#ifndef MXAO_MIPLEVEL_AO
 #define MXAO_MIPLEVEL_AO		0	//[0 to 2]      Miplevel of AO texture. 0 = fullscreen, 1 = 1/2 screen width/height, 2 = 1/4 screen width/height and so forth. Best results: IL MipLevel = AO MipLevel + 2
#endif

#ifndef MXAO_MIPLEVEL_IL
 #define MXAO_MIPLEVEL_IL		0	//[0 to 4]      Miplevel of IL texture. 0 = fullscreen, 1 = 1/2 screen width/height, 2 = 1/4 screen width/height and so forth.
#endif

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef SPATIAL_FILTER
 #define SPATIAL_FILTER	       	0   //[0 or 1]      If enabled, final GI is filtered for a less noisy but also less precise result. Enabled by default.
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float RT_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 20.0;
    ui_step = 0.01;
    ui_label = "Ray Length";
	ui_tooltip = "Maximum ray length, directly affects\nthe spread radius of shadows / indirect lighing";
    ui_category = "Path Tracing";
> = 15.0;

#define RT_SIZE_SCALE 1.0
#define texcoord_scaled texcoord

uniform int RT_RAY_AMOUNT <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Ray Amount";
    ui_category = "Path Tracing";
> = 10;

uniform int RT_RAY_STEPS <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Ray Step Amount";
    ui_category = "Path Tracing";
> = 10;

uniform float RT_Z_THICKNESS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Z Thickness";
	ui_tooltip = "The shader can't know how thick objects are, since it only\nsees the side the camera faces and has to assume a fixed value.\n\nUse this parameter to remove halos around thin objects.";
    ui_category = "Path Tracing";
> = 1.0;

uniform float RT_AO_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 2.0;
    ui_step = 0.01;
    ui_label = "Ambient Occlusion Intensity";
    ui_category = "Blending";
> = 1.0;

uniform float RT_IL_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    #define second "te"
    ui_label = "Indirect Lighting Intensity";
    ui_category = "Blending";
> = 4.0;

#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 5.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 0.0;
#endif

uniform float2 RT_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Start / End";
	ui_min = 0.00; ui_max = 1.00;
	ui_tooltip = "Distance where GI starts to fade out | is completely faded out.";
    ui_category = "Blending";
> = float2(0.0, 0.5);

uniform int RT_DEBUG_VIEW <
	ui_type = "combo";
    ui_label = "Enable Debug View";
    #define fps "da"
	ui_items = "None\0AO/IL channel\0";
	ui_tooltip = "Different debug outputs";
    ui_category = "Debug";
> = 0;
/*
uniform float3 SKY_COLOR <
	ui_type = "color";
	ui_label = "Sky Color";
> = float3(1.0, 0.0, 0.0);
*/
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

/*=============================================================================
	Textures, Samplers, Globals
=============================================================================*/

#include "qUINT_common.fxh"

uniform float4 framelen < source = matrixmul(fps, second); >;
uniform int framecount < source = "framecount"; >;

texture2D MXAO_ColorTex 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA8; MipLevels = 3+MXAO_MIPLEVEL_IL;};
texture2D MXAO_DepthTex 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;  MipLevels = 3+MXAO_MIPLEVEL_AO;};
texture2D MXAO_NormalTex	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA8; MipLevels = 3+MXAO_MIPLEVEL_IL;};

texture2D GITexture	            { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; MipLevels = 4;};
texture2D GBufferTexture	    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture2D GITexturePrev	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture2D GBufferTexturePrev    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };

texture JitterTexture           < source = "LDR_RGB1_18.png"; > { Width = 32; Height = 32; Format = RGBA8; };

sampler2D sMXAO_ColorTex	{ Texture = MXAO_ColorTex;	};
sampler2D sMXAO_DepthTex	{ Texture = MXAO_DepthTex;	};
sampler2D sMXAO_NormalTex	{ Texture = MXAO_NormalTex;	};

sampler2D sGITexture	        { Texture = GITexture;	};
sampler2D sGBufferTexture	    { Texture = GBufferTexture;	};
sampler2D sGITexturePrev	    { Texture = GITexturePrev;	};
sampler2D sGBufferTexturePrev	{ Texture = GBufferTexturePrev;	};

sampler	sJitterTexture        { Texture = JitterTexture; AddressU = WRAP; AddressV = WRAP;};

/*=============================================================================
	Vertex Shader
=============================================================================*/

struct VSOUT
{
	float4                  vpos        : SV_Position;
    float2                  texcoord          : TEXCOORD0;
    //float4                  texcoord_scaled   : TEXCOORD1;
    nointerpolation float3  texcoord2viewADD : TEXCOORD2;
    nointerpolation float3  texcoord2viewMUL : TEXCOORD3;
    nointerpolation float4  view2texcoord    : TEXCOORD4;
};

VSOUT VS_RT(in uint id : SV_VertexID)
{
    VSOUT o;

    o.texcoord.x = (id == 2) ? 2.0 : 0.0;
    o.texcoord.y = (id == 1) ? 2.0 : 0.0;

   // o.texcoord_scaled = o.texcoord.xyxy; // * float4(1 / RT_SIZE_SCALE, 1 / RT_SIZE_SCALE, RT_SIZE_SCALE, RT_SIZE_SCALE);

    o.vpos = float4(o.texcoord.xy * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    o.texcoord2viewADD = float3(-1.0,-1.0,1.0);
    o.texcoord2viewMUL = float3(2.0,2.0,0.0);

#if 1
    static const float FOV = 75; //vertical FoV
    o.texcoord2viewADD = float3(-tan(radians(FOV * 0.5)).xx,1.0) * qUINT::ASPECT_RATIO.yxx;
   	o.texcoord2viewMUL = float3(-2.0 * o.texcoord2viewADD.xy,0.0);
#endif

	o.view2texcoord.xy = rcp(o.texcoord2viewMUL.xy);
    o.view2texcoord.zw = -o.texcoord2viewADD.xy * o.view2texcoord.xy;

    return o;
}

/*=============================================================================
	Functions
=============================================================================*/

struct Ray 
{
    float3 pos;
    float3 dir;
    float2 texcoord;
    float len;
};

struct MRT
{
    float4 gi   : SV_Target0;
    float4 gbuf : SV_Target1;
};

struct RTConstants
{
    float3 pos;
    float3 normal;
    float3x3 mtbn;
    int nrays;
    int nsteps;
};

float depth2dist(in float depth)
{
    return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1;
}

float3 get_position_from_texcoord(in VSOUT i)
{
    return (i.texcoord.xyx * i.texcoord2viewMUL + i.texcoord2viewADD) * depth2dist(qUINT::linear_depth(i.texcoord.xy));
}

float3 get_position_from_texcoord(in VSOUT i, in float2 texcoord)
{
    return (texcoord.xyx * i.texcoord2viewMUL + i.texcoord2viewADD) * depth2dist(qUINT::linear_depth(texcoord));
}

float3 get_position_from_texcoord(in VSOUT i, in float2 texcoord, in int mip)
{
    return (texcoord.xyx * i.texcoord2viewMUL + i.texcoord2viewADD) * tex2Dlod(sMXAO_DepthTex, float4(texcoord.xyx, mip)).x;
}

float2 get_texcoord_from_position(in VSOUT i, in float3 pos)
{
	return (pos.xy / pos.z) * i.view2texcoord.xy + i.view2texcoord.zw;
}

float3 get_normal_from_depth(in VSOUT i)
{
    float3 d = float3(qUINT::PIXEL_SIZE, 0);
 	float3 pos = get_position_from_texcoord(i);
	float3 ddx1 = -pos + get_position_from_texcoord(i, i.texcoord.xy + d.xz);
	float3 ddx2 = pos - get_position_from_texcoord(i, i.texcoord.xy - d.xz);
	float3 ddy1 = -pos + get_position_from_texcoord(i, i.texcoord.xy + d.zy);
	float3 ddy2 = pos - get_position_from_texcoord(i, i.texcoord.xy - d.zy);

    ddx1 = abs(ddx1.z) > abs(ddx2.z) ? ddx2 : ddx1;
    ddy1 = abs(ddy1.z) > abs(ddy2.z) ? ddy2 : ddy1;

    float3 n = cross(ddy1, ddx1);
    n *= rsqrt(dot(n, n));
    return n;
}

float3x3 get_tbn(float3 n)
{
    float3 temp = float3(0.707,0.707,0);
	temp = lerp(temp, temp.zxy, saturate(1 - 10 * dot(temp, n)));
	float3 t = normalize(cross(temp, n));
	float3 b = cross(n,t);
	return float3x3(t,b,n);
}

void unpack_hdr(inout float3 color)
{
    color /= 1.01 - color; //min(min(color.r, color.g), color.b);//max(max(color.r, color.g), color.b);
}

void pack_hdr(inout float3 color)
{
    color /= 1.01 + color; //min(min(color.r, color.g), color.b);//max(max(color.r, color.g), color.b);
}

float compute_temporal_coherence(MRT curr, MRT prev)
{
    float4 gbuf_delta = abs(curr.gbuf - prev.gbuf);

    float coherence = exp(-dot(gbuf_delta.xyz, gbuf_delta.xyz) * 10)
                    * exp(-gbuf_delta.w * 5000);

    coherence = saturate(1 - coherence);
    return lerp(0.03, 0.9, coherence);
}

float4 get_spatiotemporal_jitter(in VSOUT i)
{
    float4 jitter;
    jitter.xyz = tex2Dfetch(sJitterTexture, int4(i.vpos.xy % tex2Dsize(sJitterTexture), 0, 0)).xyz;
    //reduce framecount range to minimize floating point errors
    jitter.xyz += (framecount  % 1000) * 3.1;
    jitter.xyz = frac(jitter.xyz);
    jitter.w = dot(framelen.xyz, float3(1.0, 0.07686395, 0.0024715097)) - 0x7E3 - 0.545;    
    return jitter;
}
/*=============================================================================
	Pixel Shaders
=============================================================================*/

void PS_InputBufferSetup(in VSOUT v, out float4 color : SV_Target0, out float4 depth : SV_Target1, out float4 normal : SV_Target2)
{
	normal  = float4(get_normal_from_depth(v) * 0.5 + 0.5, 1);
    color 	= tex2D(qUINT::sBackBufferTex, v.texcoord);
	depth 	= depth2dist(qUINT::linear_depth(v.texcoord));   
}

void PS_StencilSetup(in VSOUT v, out float4 o : SV_Target0)
{   
    o = float4(tex2D(sMXAO_NormalTex, v.texcoord_scaled.xy).xyz * 2 - 1, depth2dist(qUINT::linear_depth(v.texcoord.xy)) / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

    if(qUINT::linear_depth(v.texcoord.xy) >= max(RT_FADE_DEPTH.x, RT_FADE_DEPTH.y) //theoretically only .y but users might swap it...
    ) discard;    
}

void PS_RTMain(in VSOUT v, out float4 o : SV_Target0)
{
    RTConstants rtconstants;
    rtconstants.pos     = get_position_from_texcoord(v, v.texcoord_scaled.xy);
    rtconstants.normal  = tex2D(sMXAO_NormalTex, v.texcoord_scaled.xy).xyz * 2 - 1;
    rtconstants.mtbn    = get_tbn(rtconstants.normal);
    rtconstants.nrays   = RT_RAY_AMOUNT;
    rtconstants.nsteps  = RT_RAY_STEPS;  

    float4      jitter   = get_spatiotemporal_jitter(v); 

    float depth = rtconstants.pos.z / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    rtconstants.pos = rtconstants.pos * 0.998 + rtconstants.normal * depth;

    float2 sample_dir;
    sincos(38.39941 * jitter.x * saturate(1 - jitter.w*jitter.w * 300), 
    sample_dir.x, sample_dir.y); //2.3999632 * 16 

    MRT curr, prev;
    curr.gbuf = float4(rtconstants.normal, depth);  
    prev.gi = tex2D(sGITexturePrev, v.texcoord.xy);
    prev.gbuf = tex2D(sGBufferTexturePrev, v.texcoord.xy); 
    float alpha = compute_temporal_coherence(curr, prev);

    rtconstants.nrays += 15 * alpha + jitter.w * 1300;

    curr.gi = 0;

    float invthickness = 1.0 / (RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS * RT_Z_THICKNESS);    

    [loop]
    for(float r = 0; r < rtconstants.nrays; r++)
    {
        Ray ray;        
        ray.dir.z = (r + jitter.y) / rtconstants.nrays;
        ray.dir.xy = sample_dir * sqrt(1 - ray.dir.z * ray.dir.z);
        ray.dir = mul(ray.dir, rtconstants.mtbn);
        ray.len = RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS;

        sample_dir = mul(sample_dir, float2x2(0.76465, -0.64444, 0.64444, 0.76465)); 

        float intersected = 0, mip = 0; int s = 0; bool inside_screen = 1;

        while(s++ < rtconstants.nsteps && inside_screen)
        {
            float lambda = float(s - jitter.z) / rtconstants.nsteps; //normalized position in ray [0, 1]
            lambda *= lambda * rsqrt(lambda); //lambda ^ 1.5 using the fastest instruction sets

            ray.pos = rtconstants.pos + ray.dir * lambda * ray.len;

            ray.texcoord = get_texcoord_from_position(v, ray.pos);
            inside_screen = all(saturate(-ray.texcoord * ray.texcoord + ray.texcoord));

            mip = length((ray.texcoord - v.texcoord.xy) * qUINT::ASPECT_RATIO.yx) * 28;
            float3 delta = get_position_from_texcoord(v, ray.texcoord, mip + MXAO_MIPLEVEL_AO) - ray.pos;
            
            delta *= invthickness;

            [branch]
            if(delta.z < 0 && delta.z > -1 + jitter.w * 6)
            {                
                intersected = saturate(1 - dot(delta, delta)) * inside_screen; 
                s = rtconstants.nsteps;
            }
        }         

        curr.gi.w += intersected;        

        [branch]
        if(RT_IL_AMOUNT > 0 && intersected > 0.05)
        {
            float3 albedo 			= tex2Dlod(sMXAO_ColorTex, 	float4(ray.texcoord, 0, mip + MXAO_MIPLEVEL_IL)).rgb; unpack_hdr(albedo);
            float3 intersect_normal = tex2Dlod(sMXAO_NormalTex, float4(ray.texcoord, 0, mip + MXAO_MIPLEVEL_IL)).xyz * 2 - 1;

 #if INFINITE_BOUNCES != 0
            float3 nextbounce 		= tex2Dlod(sGITexturePrev, float4(ray.texcoord, 0, 0)).rgb; unpack_hdr(nextbounce);            
            albedo += nextbounce * RT_IL_BOUNCE_WEIGHT;
#endif
            curr.gi.rgb += albedo * intersected * saturate(dot(-intersect_normal, ray.dir));
        }
    }

    curr.gi /= rtconstants.nrays; 
    pack_hdr(curr.gi.rgb);

    o = lerp(prev.gi, curr.gi, alpha);
}

void PS_CopyAndFilter(in VSOUT v, out MRT o)
{
	o.gi    = tex2D(sGITexture,      v.texcoord.xy);
    o.gbuf  = tex2D(sGBufferTexture, v.texcoord.xy);
#if SPATIAL_FILTER != 0
    float jitter = dot(floor(v.vpos.xy % 4 + 0.1), float2(0.0625, 0.25)) + 0.0625;
    float2 dir; sincos(2.3999632 * 16 * jitter, dir.x, dir.y);

    float4 gi = o.gi;
    float weightsum = 1;

    [loop]
    for(int j = 0; j < 16; j++)
    {
        float2 sample_texcoord = v.texcoord.xy + dir * qUINT::PIXEL_SIZE * sqrt(j + jitter);   
        dir.xy = mul(dir.xy, float2x2(0.76465, -0.64444, 0.64444, 0.76465)); 

        float4 gi_tap   = tex2Dlod(sGITexture,      float4(sample_texcoord, 0, 0));
        float4 gbuf_tap = tex2Dlod(sGBufferTexture, float4(sample_texcoord, 0, 0));
        
        float4 gi_delta     = abs(gi_tap - o.gi);
        float4 gbuf_delta   = abs(gbuf_tap - o.gbuf);

        float ddepth    = gbuf_delta.w;
        float dnormal   = max(max(gbuf_delta.x, gbuf_delta.y), gbuf_delta.z);
        float dvalue    = dot(gi_delta, gi_delta);

        float w = exp(-ddepth * 750)
                * exp(-dnormal * 50 * saturate(ddepth * 10000))
                * exp(-dvalue * 10);

        gi += gi_tap * w;
        weightsum += w;
    }

    gi /= weightsum;
    o.gi = gi;
#endif 
}

//need this as backbuffer is not guaranteed to have RGBA8
void PS_Output(in VSOUT v, out float4 o : SV_Target0)
{
    float4 gi = tex2D(sGITexturePrev, v.texcoord.xy);
    float3 color = tex2D(sMXAO_ColorTex, v.texcoord.xy).rgb;

    unpack_hdr(color);
    unpack_hdr(gi.rgb);

    gi *= smoothstep(RT_FADE_DEPTH.y, RT_FADE_DEPTH.x, qUINT::linear_depth(v.texcoord.xy));

    gi.w = RT_AO_AMOUNT > 1 ? pow(1 - gi.w, RT_AO_AMOUNT) : 1 - gi.w * RT_AO_AMOUNT;
    gi.rgb *= RT_IL_AMOUNT * RT_IL_AMOUNT;

    color = color * gi.w * (1 + gi.rgb);  

    if(RT_DEBUG_VIEW == 1)
        color.rgb = gi.w * (1 + gi.rgb);

    pack_hdr(color.rgb);
    o = float4(color, 1);
}

/*=============================================================================
	Techniques
=============================================================================*/



technique RT
{
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_InputBufferSetup;
		RenderTarget0 = MXAO_ColorTex;
		RenderTarget1 = MXAO_DepthTex;
		RenderTarget2 = MXAO_NormalTex;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_StencilSetup;
        RenderTarget = GBufferTexture;
        ClearRenderTargets = true;
		StencilEnable = true;
		StencilPass = REPLACE;
        StencilRef = 1;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_RTMain;
        RenderTarget = GITexture;        
        ClearRenderTargets = true;
        StencilEnable = true;
        StencilPass = KEEP;
        StencilFunc = EQUAL;
        StencilRef = 1;
	}
        pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_CopyAndFilter;
        RenderTarget0 = GITexturePrev;
        RenderTarget1 = GBufferTexturePrev;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Output;
	}
}
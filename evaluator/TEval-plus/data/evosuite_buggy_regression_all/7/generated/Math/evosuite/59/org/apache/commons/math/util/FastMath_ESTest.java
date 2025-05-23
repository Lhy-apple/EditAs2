/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:57:32 GMT 2023
 */

package org.apache.commons.math.util;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.math.util.FastMath;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FastMath_ESTest extends FastMath_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      double double0 = FastMath.acos(0.8551157907223507);
      assertEquals(0.5450221825742618, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      double double0 = FastMath.toRadians(1.157335463434598E19);
      assertEquals(2.0199314387028176E17, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      double double0 = FastMath.toDegrees((-1.0));
      assertEquals((-57.29577951308232), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      double double0 = FastMath.expm1((-744.0));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      double double0 = FastMath.ulp((-1723.0243108725147));
      assertEquals(2.2737367544323206E-13, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      double double0 = FastMath.tan(2127181744);
      assertEquals((-0.6133309462233124), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      long long0 = FastMath.round((-1.0));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      int int0 = FastMath.round(0.0F);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      double double0 = FastMath.cos(1.9137534259255532E16);
      assertEquals((-0.9420721322943734), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      double double0 = FastMath.random();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      double double0 = FastMath.log10(0.0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      double double0 = FastMath.atan((-744.0));
      assertEquals((-1.5694522415827843), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      double double0 = FastMath.acosh(1.157335463434598E19);
      assertEquals(44.5883842961004, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      double double0 = FastMath.nextUp((-4.503599627370496E15));
      assertEquals((-4.5035996273704955E15), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      double double0 = FastMath.cosh((-877.538972289));
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      double double0 = FastMath.cosh(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      double double0 = FastMath.cosh(1.157335463434598E19);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      double double0 = FastMath.cosh(0.0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      double double0 = FastMath.cosh((-0.1428571423679182));
      assertEquals(1.0102214472525175, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      double double0 = FastMath.sinh(4.9E-324);
      assertEquals(4.9E-324, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      double double0 = FastMath.sinh(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      double double0 = FastMath.sinh(417.27);
      assertEquals(8.260920983346568E180, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      double double0 = FastMath.sinh((-302932621132653718L));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      double double0 = FastMath.sinh(-0.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      double double0 = FastMath.sinh((-1.0));
      assertEquals((-1.1752011936438014), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      double double0 = FastMath.tanh(0.5169898643703855);
      assertEquals(0.47537344292280326, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      double double0 = FastMath.tanh(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      double double0 = FastMath.tanh(2474.13309158);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      double double0 = FastMath.tanh((-408.65279808374));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      double double0 = FastMath.tanh(-0.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      double double0 = FastMath.tanh((-20.0));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      double double0 = FastMath.tanh((-3.301064098877953E-18));
      assertEquals((-3.301064098877953E-18), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      double double0 = FastMath.asinh(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      double double0 = FastMath.asinh((-349.7265319));
      assertEquals((-6.550300736214586), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      double double0 = FastMath.asinh(0.08371849358081818);
      assertEquals(0.08362100656848195, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      double double0 = FastMath.asinh(0.15);
      assertEquals(0.14944312018495765, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      double double0 = FastMath.asinh(0.009405819475636835);
      assertEquals(0.009405680793227579, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      double double0 = FastMath.asinh(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      double double0 = FastMath.atanh(0.9999999999999999);
      assertEquals(18.714973875118524, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      double double0 = FastMath.atanh((-0.11820594773372614));
      assertEquals((-0.1187611598470173), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      double double0 = FastMath.atanh((-0.05417713522911072));
      assertEquals((-0.05423023499691139), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      double double0 = FastMath.atanh(0.031);
      assertEquals(0.031009936063096846, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      double double0 = FastMath.atanh(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      double double0 = FastMath.signum(-0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      double double0 = FastMath.signum((-11.749571813298024));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      double double0 = FastMath.signum(469.961916071954);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      double double0 = FastMath.signum(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      double double0 = FastMath.pow((-8.0E298), (-8.0E298));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      double double0 = FastMath.expm1((-1964.0));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      double double0 = FastMath.pow(0.2857142686843872, 594.550909976);
      assertEquals(4.9E-324, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      double double0 = FastMath.pow(1.570796326795, (-1570.1032456151095));
      assertEquals(1.179083454793476E-308, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      double double0 = FastMath.expm1(5478.625801551925);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      double double0 = FastMath.expm1(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      double double0 = FastMath.expm1(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      double double0 = FastMath.expm1((-0.31934582649321014));
      assertEquals((-0.27337578005519786), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      double double0 = FastMath.atanh(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      double double0 = FastMath.log(-0.0);
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      double double0 = FastMath.log1p((-20.0));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      double double0 = FastMath.log10(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      double double0 = FastMath.pow(4.9E-324, 4.9E-324);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      double double0 = FastMath.pow(0.99, 0.99);
      assertEquals(0.9900995033250751, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      double double0 = FastMath.log(1.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      double double0 = FastMath.pow((-1.0), (-1.0));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      double double0 = FastMath.log1p((-1.0));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      double double0 = FastMath.log1p(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      double double0 = FastMath.log1p(3.141592653589793);
      assertEquals(1.4210804127942926, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      double double0 = FastMath.log1p(-0.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      double double0 = FastMath.pow(0.0, 0.0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      double double0 = FastMath.pow(Double.NaN, (-1));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, (-1964.0));
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      double double0 = FastMath.pow(0.0, 1.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, (-2728.8836));
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, (-1.0));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, 16.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, 1763.671);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      double double0 = FastMath.pow(-0.0, 1.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      double double0 = FastMath.pow(Double.POSITIVE_INFINITY, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      double double0 = FastMath.pow(Double.POSITIVE_INFINITY, (-1027.0688208963054));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      double double0 = FastMath.pow(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      double double0 = FastMath.pow(1.0, Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      double double0 = FastMath.pow(5.400932872855145E-17, Double.POSITIVE_INFINITY);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      double double0 = FastMath.pow(1819.2, Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, 4.9E-324);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, (-302932621132653736L));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, (-4247.0));
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, 2987.0);
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      double double0 = FastMath.pow(Double.NEGATIVE_INFINITY, 20.0);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      double double0 = FastMath.pow((-1.0), Double.NEGATIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      double double0 = FastMath.pow((-3.301064098877953E-18), Double.NEGATIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      double double0 = FastMath.pow((-302932621132653736L), Double.NEGATIVE_INFINITY);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      double double0 = FastMath.pow((-515.7964403839), 1.157335463434598E19);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      double double0 = FastMath.pow((-816.3462441963377), (-816.3462441963377));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      double double0 = FastMath.pow((-20.0), (-20.0));
      assertEquals(9.5367431640625E-27, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      double double0 = FastMath.pow(44.3614195558365, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      double double0 = FastMath.sin(1.0);
      assertEquals(0.8414709848078965, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      double double0 = FastMath.tan(1.157335463434598E19);
      assertEquals((-0.026041725054373393), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      double double0 = FastMath.tan((-1.0));
      assertEquals((-1.5574077246549023), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      double double0 = FastMath.cos(5.080151458968558E20);
      assertEquals((-0.8690549174523372), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      double double0 = FastMath.sin((-302932621132653718L));
      assertEquals((-0.770703077671632), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      double double0 = FastMath.sin(4577762542105553359L);
      assertEquals((-0.06720747742781692), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      double double0 = FastMath.cos(2.2654367799600815E10);
      assertEquals(0.9003201594114619, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      double double0 = FastMath.sin((-2146996248));
      assertEquals(0.8731179300556317, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      double double0 = FastMath.cos((-302932621132653740L));
      assertEquals((-0.9587561512775762), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      double double0 = FastMath.sin(-0.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      double double0 = FastMath.sin(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      double double0 = FastMath.sin(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      double double0 = FastMath.sin(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      double double0 = FastMath.sin(3.141592653589793);
      assertEquals(1.2246467991473532E-16, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      double double0 = FastMath.cos(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      double double0 = FastMath.cos(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      double double0 = FastMath.cos(9.313225746154787E-10);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      double double0 = FastMath.cos(1398.7);
      assertEquals((-0.7704375312895825), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      double double0 = FastMath.cos((-3.141592653589793));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      double double0 = FastMath.tan(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      double double0 = FastMath.tan(-0.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
      double double0 = FastMath.tan(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test119()  throws Throwable  {
      double double0 = FastMath.tan(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      double double0 = FastMath.tan((-1033.6873063413702));
      assertEquals((-0.1036925703635692), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      double double0 = FastMath.tan(3.141592653589793);
      assertEquals((-1.2246467991473532E-16), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      double double0 = FastMath.atan(1.157335463434598E19);
      assertEquals(1.5707963267948966, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test123()  throws Throwable  {
      double double0 = FastMath.atan2((-302932621132653744L), 2.718281828459045);
      assertEquals((-1.5707963267948966), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test124()  throws Throwable  {
      double double0 = FastMath.atan2((-4.9E-324), 1.5704175587624856);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test125()  throws Throwable  {
      double double0 = FastMath.atan2(1.4177906801102105, (-2.883492271129372E300));
      assertEquals(3.141592653589793, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test126()  throws Throwable  {
      double double0 = FastMath.atan2(2.5588580733482136E-17, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test127()  throws Throwable  {
      double double0 = FastMath.atan2(Double.NaN, (-302932621132653749L));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test128()  throws Throwable  {
      double double0 = FastMath.atan2(-0.0, Double.POSITIVE_INFINITY);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test129()  throws Throwable  {
      double double0 = FastMath.atan2(-0.0, -0.0);
      assertEquals((-3.141592653589793), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test130()  throws Throwable  {
      double double0 = FastMath.atan2(0.0, Double.NEGATIVE_INFINITY);
      assertEquals(3.141592653589793, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test131()  throws Throwable  {
      double double0 = FastMath.atan2(0.0, (-3899.6339));
      assertEquals(3.141592653589793, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test132()  throws Throwable  {
      double double0 = FastMath.atan2(0.0, 0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test133()  throws Throwable  {
      double double0 = FastMath.atan2(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
      assertEquals(2.356194490192345, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test134()  throws Throwable  {
      double double0 = FastMath.atan2(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
      assertEquals(0.7853981633974483, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test135()  throws Throwable  {
      double double0 = FastMath.atan2(Double.POSITIVE_INFINITY, (-1437.4239937266743));
      assertEquals(1.5707963267948966, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test136()  throws Throwable  {
      double double0 = FastMath.atan2(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
      assertEquals((-0.7853981633974483), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test137()  throws Throwable  {
      double double0 = FastMath.atan2(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
      assertEquals((-2.356194490192345), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test138()  throws Throwable  {
      double double0 = FastMath.atan2(Double.NEGATIVE_INFINITY, (-302932621132653743L));
      assertEquals((-1.5707963267948966), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test139()  throws Throwable  {
      double double0 = FastMath.atan2((-1.0), Double.POSITIVE_INFINITY);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test140()  throws Throwable  {
      double double0 = FastMath.atan2(5478.625801551925, Double.POSITIVE_INFINITY);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test141()  throws Throwable  {
      double double0 = FastMath.atan2(1.5707963267948966, Double.NEGATIVE_INFINITY);
      assertEquals(3.141592653589793, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test142()  throws Throwable  {
      double double0 = FastMath.atan2((-302932621132653753L), Double.NEGATIVE_INFINITY);
      assertEquals((-3.141592653589793), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test143()  throws Throwable  {
      double double0 = FastMath.atan2((-574.432), 0.0);
      assertEquals((-1.5707963267948966), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test144()  throws Throwable  {
      double double0 = FastMath.atan2(8.0E298, 0.0);
      assertEquals(1.5707963267948966, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test145()  throws Throwable  {
      double double0 = FastMath.atan2((-1872.6317273), 1.0766676266880148E299);
      assertEquals((-1.7392848831728003E-296), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test146()  throws Throwable  {
      double double0 = FastMath.asin(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test147()  throws Throwable  {
      double double0 = FastMath.asin(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test148()  throws Throwable  {
      double double0 = FastMath.asin(-0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test149()  throws Throwable  {
      double double0 = FastMath.asin((-32.098987422025516));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test150()  throws Throwable  {
      double double0 = FastMath.asin(1.0);
      assertEquals(1.5707963267948966, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test151()  throws Throwable  {
      double double0 = FastMath.asin((-1));
      assertEquals((-1.5707963267948966), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test152()  throws Throwable  {
      double double0 = FastMath.acos(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test153()  throws Throwable  {
      double double0 = FastMath.acos(1620.0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test154()  throws Throwable  {
      double double0 = FastMath.acos((-302932621132653753L));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test155()  throws Throwable  {
      double double0 = FastMath.acos((-1.0));
      assertEquals(3.141592653589793, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test156()  throws Throwable  {
      double double0 = FastMath.acos(1.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test157()  throws Throwable  {
      double double0 = FastMath.acos(0.0);
      assertEquals(1.5707963267948966, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test158()  throws Throwable  {
      double double0 = FastMath.acos((-4.9E-324));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test159()  throws Throwable  {
      double double0 = FastMath.cbrt((-829.4));
      assertEquals((-9.39553129981705), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test160()  throws Throwable  {
      double double0 = FastMath.cbrt((-4.9E-324));
      assertEquals((-1.7031839360032603E-108), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test161()  throws Throwable  {
      double double0 = FastMath.cbrt(-0.0);
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test162()  throws Throwable  {
      double double0 = FastMath.cbrt(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test163()  throws Throwable  {
      int int0 = FastMath.abs(0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test164()  throws Throwable  {
      int int0 = FastMath.abs((-2179));
      assertEquals(2179, int0);
  }

  @Test(timeout = 4000)
  public void test165()  throws Throwable  {
      long long0 = FastMath.abs(4485585228861014016L);
      assertEquals(4485585228861014016L, long0);
  }

  @Test(timeout = 4000)
  public void test166()  throws Throwable  {
      long long0 = FastMath.abs((-302932621132653753L));
      assertEquals(302932621132653753L, long0);
  }

  @Test(timeout = 4000)
  public void test167()  throws Throwable  {
      float float0 = FastMath.abs(1.0F);
      assertEquals(1.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test168()  throws Throwable  {
      float float0 = FastMath.abs((-676.51F));
      assertEquals(676.51F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test169()  throws Throwable  {
      double double0 = FastMath.abs(2.2654367781713455E10);
      assertEquals(2.2654367781713455E10, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test170()  throws Throwable  {
      double double0 = FastMath.nextAfter(Double.NaN, (-3128.031));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test171()  throws Throwable  {
      double double0 = FastMath.nextAfter(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test172()  throws Throwable  {
      double double0 = FastMath.nextAfter(0.0, 1398.7);
      assertEquals(4.9E-324, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test173()  throws Throwable  {
      double double0 = FastMath.nextAfter(-0.0, (-1964.0));
      assertEquals((-4.9E-324), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test174()  throws Throwable  {
      double double0 = FastMath.nextAfter(9.313225746154785E-10, 9.313225746154785E-10);
      assertEquals(9.313225746154787E-10, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test175()  throws Throwable  {
      double double0 = FastMath.nextAfter((-0.49999999999999994), (-408.751635368369));
      assertEquals((-0.5), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test176()  throws Throwable  {
      double double0 = FastMath.nextAfter((-0.7704375312895825), 0.0);
      assertEquals((-0.7704375312895824), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test177()  throws Throwable  {
      double double0 = FastMath.rint(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test178()  throws Throwable  {
      double double0 = FastMath.rint(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test179()  throws Throwable  {
      double double0 = FastMath.rint(Double.NEGATIVE_INFINITY);
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test180()  throws Throwable  {
      double double0 = FastMath.ceil(1916.00219941);
      assertEquals(1917.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test181()  throws Throwable  {
      double double0 = FastMath.ceil((-1));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test182()  throws Throwable  {
      double double0 = FastMath.rint(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test183()  throws Throwable  {
      double double0 = FastMath.ceil(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test184()  throws Throwable  {
      double double0 = FastMath.ceil((-6.483922519250713E-18));
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test185()  throws Throwable  {
      double double0 = FastMath.rint(225.7275382);
      assertEquals(226.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test186()  throws Throwable  {
      int int0 = FastMath.min(18, (-280106563));
      assertEquals((-280106563), int0);
  }

  @Test(timeout = 4000)
  public void test187()  throws Throwable  {
      int int0 = FastMath.min((-2145837005), 2179);
      assertEquals((-2145837005), int0);
  }

  @Test(timeout = 4000)
  public void test188()  throws Throwable  {
      long long0 = FastMath.min(4503599627370495L, (-1L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test189()  throws Throwable  {
      long long0 = FastMath.min((-302932621132653753L), (-302932621132653753L));
      assertEquals((-302932621132653753L), long0);
  }

  @Test(timeout = 4000)
  public void test190()  throws Throwable  {
      float float0 = FastMath.min(63.0F, (-549.9744F));
      assertEquals((-549.9744F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test191()  throws Throwable  {
      float float0 = FastMath.min(57.9F, 57.9F);
      assertEquals(57.9F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test192()  throws Throwable  {
      float float0 = FastMath.min((-1271.829F), Float.NaN);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test193()  throws Throwable  {
      double double0 = FastMath.min(Double.NaN, 0.5);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test194()  throws Throwable  {
      double double0 = FastMath.min(0.0, 0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test195()  throws Throwable  {
      double double0 = FastMath.min(Double.POSITIVE_INFINITY, 1174.075013455584);
      assertEquals(1174.075013455584, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test196()  throws Throwable  {
      int int0 = FastMath.max(1, (-2146622048));
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test197()  throws Throwable  {
      int int0 = FastMath.max((-2193), (-2193));
      assertEquals((-2193), int0);
  }

  @Test(timeout = 4000)
  public void test198()  throws Throwable  {
      long long0 = FastMath.max(1469L, 596L);
      assertEquals(1469L, long0);
  }

  @Test(timeout = 4000)
  public void test199()  throws Throwable  {
      long long0 = FastMath.max((-302932621132653753L), (-302932621132653753L));
      assertEquals((-302932621132653753L), long0);
  }

  @Test(timeout = 4000)
  public void test200()  throws Throwable  {
      float float0 = FastMath.max(27.6F, (-1819.0F));
      assertEquals((-1819.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test201()  throws Throwable  {
      float float0 = FastMath.max((-676.51F), (-676.51F));
      assertEquals((-676.51F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test202()  throws Throwable  {
      float float0 = FastMath.max(Float.NaN, 1433.0F);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test203()  throws Throwable  {
      double double0 = FastMath.max(0.0, (-4.9E-324));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test204()  throws Throwable  {
      double double0 = FastMath.max(1.8369957088341409, 1.8369957088341409);
      assertEquals(1.8369957088341409, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test205()  throws Throwable  {
      double double0 = FastMath.max(Double.NaN, (-1.5165926535897931));
      assertEquals(Double.NaN, double0, 0.01);
  }
}

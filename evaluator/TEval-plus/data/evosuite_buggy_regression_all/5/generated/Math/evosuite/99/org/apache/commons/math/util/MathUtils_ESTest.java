/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:37:05 GMT 2023
 */

package org.apache.commons.math.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.util.MathUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MathUtils_ESTest extends MathUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      int int0 = MathUtils.hash((double) 1073741824);
      assertEquals(1104150528, int0);
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      int int0 = MathUtils.hash(doubleArray0);
      assertEquals(887503681, int0);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      float float0 = MathUtils.round((-879.0F), 1, 1);
      assertEquals((-878.9F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck(9218868437227405312L, 9218868437227405312L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      double double0 = MathUtils.sinh((-3433.9537));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      double double0 = MathUtils.normalizeAngle((-2131.934747877), 0.0);
      assertEquals((-1.9349287431205084), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      float float0 = MathUtils.round((-1269.536F), (-1));
      assertEquals((-1270.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      double double0 = MathUtils.cosh(0.0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      double double0 = MathUtils.round(843.0, 20);
      assertEquals(843.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      double double0 = MathUtils.log(0.0, 0.0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck((-1073741826), (-1073741826));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      int int0 = MathUtils.addAndCheck((-1073741824), (-1073741824));
      assertEquals(Integer.MIN_VALUE, int0);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck(2146941519, 2146941519);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      long long0 = MathUtils.subAndCheck(3960L, 3960L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck((-9223372036854775529L), (-9223372036854775529L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      long long0 = MathUtils.addAndCheck((-266L), (-1862L));
      assertEquals((-2128L), long0);
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      long long0 = MathUtils.addAndCheck(1688L, 1688L);
      assertEquals(3376L, long0);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(14, 0);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient((-2147483646), 1871);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient((-370), (-1073741824));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(6, 6);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(67, 1);
      assertEquals(67L, long0);
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(0, (-1));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(22, 15);
      assertEquals(170544L, long0);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(66, 6);
      assertEquals(90858768L, long0);
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(1974, 6);
      assertEquals(81554365709736381L, long0);
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(7, 5);
      assertEquals(21.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientDouble(3, 5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientDouble((-1073741824), (-1073741824));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(15, 15);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(13, 0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(1075, 1);
      assertEquals(1075.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(0, (-1));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(958, 425);
      assertEquals(1.4163790249197623E284, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientLog((-1), 5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientLog((-1073741824), (-1073741824));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(667, 667);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1, 0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1520, 1);
      assertEquals(7.326465613840322, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(0, (-1));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(61, 20);
      assertEquals(36.36921904562971, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(475, (-332));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(2075, 1058);
      assertEquals(1433.830738193802, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) Float.NaN, (double) Float.NaN);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      boolean boolean0 = MathUtils.equals(Double.NaN, 1.0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) 16, (double) 16, (double) 16);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) 1L, (double) 3, (double) 13);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) 20, (double) (-1183L), Double.POSITIVE_INFINITY);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) 10, 1215.0, (-1.0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      boolean boolean0 = MathUtils.equals(2.45314165264E11, (double) 228, (double) 9);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double[]) null, (double[]) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      boolean boolean0 = MathUtils.equals(doubleArray0, (double[]) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      boolean boolean0 = MathUtils.equals((double[]) null, doubleArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      double[] doubleArray1 = new double[1];
      boolean boolean0 = MathUtils.equals(doubleArray1, doubleArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      double[] doubleArray1 = new double[7];
      doubleArray1[1] = (-3494.4888);
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorial((-4));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorial(67);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // factorial value is too large to fit in a long
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      double double0 = MathUtils.factorialDouble(1783);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorialDouble((-1789));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      double double0 = MathUtils.factorialDouble(16);
      assertEquals(2.0922789888E13, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorialLog((-1073741839));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n > 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      double double0 = MathUtils.factorialLog(1);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      int int0 = MathUtils.gcd(0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      int int0 = MathUtils.gcd(541, 0);
      assertEquals(541, int0);
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.gcd(Integer.MIN_VALUE, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: gcd(-2,147,483,648, -2,147,483,648) is 2^31
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      byte byte0 = MathUtils.indicator((byte) (-20));
      assertEquals((byte) (-1), byte0);
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      byte byte0 = MathUtils.indicator((byte)0);
      assertEquals((byte)1, byte0);
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      double double0 = MathUtils.indicator((-4784.574931559));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      double double0 = MathUtils.indicator(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      double double0 = MathUtils.indicator(108.0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 6, 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      float float0 = MathUtils.round(0.0F, (-1), 5);
      assertEquals(-0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      int int0 = MathUtils.indicator((-1073741819));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      int int0 = MathUtils.indicator((int) (byte)1);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      long long0 = MathUtils.indicator((long) (byte) (-73));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      long long0 = MathUtils.indicator(1688L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      short short0 = MathUtils.indicator((short) (-1));
      assertEquals((short) (-1), short0);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      short short0 = MathUtils.indicator((short)1855);
      assertEquals((short)1, short0);
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      int int0 = MathUtils.lcm(0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.lcm((-1073741824), 18);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: mul
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      int int0 = MathUtils.lcm((-3185), 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      int int0 = MathUtils.mulAndCheck(5, 13);
      assertEquals(65, int0);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((-1073741816), (-1073741816));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: mul
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((long) (-2147482096), (long) 856);
      assertEquals((-1838244674176L), long0);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((-306L), (-34674330965619457L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((-1862L), (-237L));
      assertEquals(441294L, long0);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((long) (-2147482952), 0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((long) 1563, (-9218868437227405313L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck(0L, 0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient(256, 16);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      float float0 = MathUtils.round((float) 3960L, 1795);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      float float0 = MathUtils.round((float) (-262075), (-65350), 3);
      assertEquals(Float.NEGATIVE_INFINITY, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      double double0 = MathUtils.nextAfter((-0.9999999999999999), (-1.15292343833324493E18));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      double double0 = MathUtils.nextAfter((-1.0), 1073741824);
      assertEquals((-0.9999999999999999), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      double double0 = MathUtils.scalb(0, 2146230146);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      double double0 = MathUtils.scalb(1712.187278539, 3011);
      assertEquals((-1.334864214054956E293), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      double double0 = MathUtils.scalb(Float.NaN, (-1));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      double double0 = MathUtils.scalb(Double.POSITIVE_INFINITY, (-988));
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      double double0 = MathUtils.round((double) Float.NaN, (-2095), 2734);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      double double0 = MathUtils.round(Double.NEGATIVE_INFINITY, 0, 4);
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      float float0 = MathUtils.round((float) 0, 541, 0);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      float float0 = MathUtils.round((float) 29, 2, 2);
      assertEquals(29.01F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      float float0 = MathUtils.round(1.0F, 7, 7);
      assertEquals(1.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.round((float) 8, 8, 8);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid rounding method.
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      float float0 = MathUtils.round((float) (-1183L), 9, 2);
      assertEquals((-1183.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      float float0 = MathUtils.round(21.0F, 3, 3);
      assertEquals(20.999F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      float float0 = MathUtils.round(1824.993F, 2472, 5);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      float float0 = MathUtils.round((-1566.6F), (-1), 6);
      assertEquals((-1570.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      float float0 = MathUtils.round(2642.2268F, 6, 6);
      assertEquals(2642.2268F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      float float0 = MathUtils.round((float) 6, 541, 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.round((float) 2472, (-9), 7);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Inexact result from rounding
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte)125);
      assertEquals((byte)1, byte0);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte)0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte) (-73));
      assertEquals((byte) (-1), byte0);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      double double0 = MathUtils.sign((double) 1215);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      double double0 = MathUtils.sign(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      double double0 = MathUtils.sign((double) 0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
      double double0 = MathUtils.sign((-516.472));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test119()  throws Throwable  {
      float float0 = MathUtils.sign((float) (-56));
      assertEquals((-1.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      float float0 = MathUtils.sign(Float.NaN);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      float float0 = MathUtils.sign(0.0F);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      float float0 = MathUtils.sign((float) 2144393742L);
      assertEquals(1.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test123()  throws Throwable  {
      int int0 = MathUtils.sign(3011);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test124()  throws Throwable  {
      int int0 = MathUtils.sign(0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test125()  throws Throwable  {
      int int0 = MathUtils.sign((-1073741824));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test126()  throws Throwable  {
      long long0 = MathUtils.sign(3739L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test127()  throws Throwable  {
      long long0 = MathUtils.sign((long) 0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test128()  throws Throwable  {
      long long0 = MathUtils.sign((-2925L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test129()  throws Throwable  {
      short short0 = MathUtils.sign((short) (byte)125);
      assertEquals((short)1, short0);
  }

  @Test(timeout = 4000)
  public void test130()  throws Throwable  {
      short short0 = MathUtils.sign((short)0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test131()  throws Throwable  {
      short short0 = MathUtils.sign((short) (-2603));
      assertEquals((short) (-1), short0);
  }

  @Test(timeout = 4000)
  public void test132()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(Integer.MIN_VALUE, 4702);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test133()  throws Throwable  {
      int int0 = MathUtils.subAndCheck(5, 13);
      assertEquals((-8), int0);
  }

  @Test(timeout = 4000)
  public void test134()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(1073741824, (-1073741824));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test135()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(2144393742L, (-9223372036854775808L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test136()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((-9223372036854775808L), (-9223372036854775808L));
      assertEquals(0L, long0);
  }
}

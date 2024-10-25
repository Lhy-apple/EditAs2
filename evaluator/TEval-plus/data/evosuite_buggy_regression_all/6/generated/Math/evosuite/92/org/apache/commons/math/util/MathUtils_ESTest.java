/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:56:39 GMT 2023
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
      int int0 = MathUtils.hash((double) 6);
      assertEquals(1075314688, int0);
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      int int0 = MathUtils.hash(doubleArray0);
      assertEquals(31, int0);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.lcm(1073741824, (-104));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: mul
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      float float0 = MathUtils.round((float) 1075314688, 1075314688, 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      long long0 = MathUtils.addAndCheck(23L, 23L);
      assertEquals(46L, long0);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      double double0 = MathUtils.sinh((-49L));
      assertEquals((-9.536732862475499E20), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      double double0 = MathUtils.normalizeAngle(2155.1644242397188, 2155.1644242397188);
      assertEquals(2155.1644242397188, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(3112, (-3767));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      float float0 = MathUtils.round((float) (-450), (-450));
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      double double0 = MathUtils.cosh(0.0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      double double0 = MathUtils.log(2.2250738585072014E-308, 1202);
      assertEquals((-0.010010979628875916), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck((-2147483645), (-2147483645));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      int int0 = MathUtils.addAndCheck((-1), (-1));
      assertEquals((-2), int0);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck(1073741837, 1073741837);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck((long) (-2147475346), 9223372036854775807L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((long) 1515, 1869L);
      assertEquals((-354L), long0);
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((-1L), 1511L);
      assertEquals((-1512L), long0);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(9223372036854775766L, (-3221L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient((-1), 8155);
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
        MathUtils.binomialCoefficient((-518), (-518));
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
      long long0 = MathUtils.binomialCoefficient(0, 0);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient((byte)1, 0);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(7, 1);
      assertEquals(7L, long0);
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient((byte)5, 4);
      assertEquals(5L, long0);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient(569, 52);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // result too large to represent in a long integer
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientLog(4, 340);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientDouble((-929), (-929));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1202, 1202);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1, 0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1515, 1);
      assertEquals(7.323170717943469, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(0, (-1));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = Double.NaN;
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      boolean boolean0 = MathUtils.equals(Double.NaN, (-3.465596652319744E7));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double[]) null, (double[]) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      boolean boolean0 = MathUtils.equals(doubleArray0, (double[]) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      boolean boolean0 = MathUtils.equals((double[]) null, doubleArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      double[] doubleArray1 = new double[0];
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (double) (byte)42;
      double[] doubleArray1 = new double[4];
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      double double0 = MathUtils.factorialLog(0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorial((-3783));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorial(3104);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // factorial value is too large to fit in a long
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      double double0 = MathUtils.factorialDouble(20);
      assertEquals(2.43290200817664E18, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorialDouble((short) (-1644));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      double double0 = MathUtils.factorialDouble(184);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorialLog((-996));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n > 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      int int0 = MathUtils.gcd(0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      int int0 = MathUtils.gcd(497, 0);
      assertEquals(497, int0);
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      int int0 = MathUtils.gcd((-2827), 16);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      byte byte0 = MathUtils.indicator((byte) (-1));
      assertEquals((byte) (-1), byte0);
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      byte byte0 = MathUtils.indicator((byte)111);
      assertEquals((byte)1, byte0);
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      double double0 = MathUtils.indicator((double) 1202);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      double double0 = MathUtils.indicator(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      double double0 = MathUtils.indicator((-2254.0357));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 5, 5);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      int int0 = MathUtils.indicator((-20));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      int int0 = MathUtils.indicator(1073741824);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      long long0 = MathUtils.indicator((-409L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      long long0 = MathUtils.indicator(23L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      short short0 = MathUtils.indicator((short) (-1));
      assertEquals((short) (-1), short0);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      short short0 = MathUtils.indicator((short)693);
      assertEquals((short)1, short0);
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      int int0 = MathUtils.lcm(1073741849, 1073741849);
      assertEquals(1073741849, int0);
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck(1073741824, 1073741824);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: mul
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((-1959L), (-2449L));
      assertEquals(4797591L, long0);
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((long) 0, (long) (byte)0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((-1959L), 121645100408832000L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((-255L), (-9223372036854775808L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((long) 0, (long) (byte) (-1));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((-1L), 3675L);
      assertEquals((-3675L), long0);
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck(355687428096021L, 355687428096021L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck(3675L, 3675L);
      assertEquals(13505625L, long0);
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      float float0 = MathUtils.round((-1556.036F), 2298, 2);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      float float0 = MathUtils.round(2769.7507F, 11, 1);
      assertEquals(2769.7507F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      double double0 = MathUtils.nextAfter(0.0, (-802.786));
      assertEquals((-4.9E-324), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      double double0 = MathUtils.nextAfter(0.9999999999999999, 618.5238);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      float float0 = MathUtils.round((-597.457F), 4, 4);
      assertEquals((-597.457F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      double double0 = MathUtils.nextAfter(1, (-2832.7009));
      assertEquals(0.9999999999999999, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      double double0 = MathUtils.scalb((short)0, 0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      double double0 = MathUtils.scalb(Float.NaN, (-1648));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      double double0 = MathUtils.scalb(Double.NEGATIVE_INFINITY, (-452));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      double double0 = MathUtils.scalb(2088.279058445, 2641);
      assertEquals((-6.769801925926303E181), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      double double0 = MathUtils.round(Double.NaN, 1);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      double double0 = MathUtils.round(Double.POSITIVE_INFINITY, 0, (-1));
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      float float0 = MathUtils.round((float) 6, 6, 0);
      assertEquals(6.000001F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      float float0 = MathUtils.round((float) 3, 3, 3);
      assertEquals(2.999F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      float float0 = MathUtils.round(2785.482F, 1073741842, 7);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.round(1.0F, (-1), (-26));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid rounding method.
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 2, 2);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      float float0 = MathUtils.round((-3936.7874F), 10, 3);
      assertEquals((-3936.787F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      float float0 = MathUtils.round((float) 5, 5, 5);
      assertEquals(5.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      float float0 = MathUtils.round(2769.7507F, (-2), 6);
      assertEquals(2800.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      float float0 = MathUtils.round((float) 6, 6, 6);
      assertEquals(6.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 6, 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.round(Float.NaN, 7, 7);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Inexact result from rounding
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte)42);
      assertEquals((byte)1, byte0);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte)0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte) (-1));
      assertEquals((byte) (-1), byte0);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      double double0 = MathUtils.sign((-1299.7311649459));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      double double0 = MathUtils.sign(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      double double0 = MathUtils.sign(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      double double0 = MathUtils.sign(3.141592653589793);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      float float0 = MathUtils.sign(1.0F);
      assertEquals(1.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      float float0 = MathUtils.sign(Float.NaN);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      float float0 = MathUtils.sign(0.0F);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      float float0 = MathUtils.sign((-1745.6F));
      assertEquals((-1.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      int int0 = MathUtils.sign(258);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      int int0 = MathUtils.sign(0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      int int0 = MathUtils.sign((-6));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      long long0 = MathUtils.sign((-1L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      long long0 = MathUtils.sign((long) (byte)0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      long long0 = MathUtils.sign(23L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      short short0 = MathUtils.sign((short) (-1644));
      assertEquals((short) (-1), short0);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      short short0 = MathUtils.sign((short)0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      short short0 = MathUtils.sign((short)693);
      assertEquals((short)1, short0);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck((-116), Integer.MAX_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      int int0 = MathUtils.subAndCheck(0, 1);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(Integer.MAX_VALUE, (-1));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((-488L), (-9223372036854775808L));
      assertEquals(9223372036854775320L, long0);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(1L, (-9223372036854775787L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }
}

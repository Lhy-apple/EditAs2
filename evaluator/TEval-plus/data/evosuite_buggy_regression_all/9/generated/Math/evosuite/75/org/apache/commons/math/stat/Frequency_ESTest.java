/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:46:25 GMT 2023
 */

package org.apache.commons.math.stat;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Comparator;
import org.apache.commons.math.stat.Frequency;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Frequency_ESTest extends Frequency_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct((Object) "(Q=?");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.clear();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer(9);
      frequency0.addValue(9);
      long long0 = frequency0.getCumFreq((Object) integer0);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct((Object) "GsU[m*B f4^\"gv");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct(1L);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCount('$');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCumFreq('*');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct((-826));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue('F');
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((-51));
      frequency0.addValue((-1));
      long long0 = frequency0.getCumFreq((-41));
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer(0);
      long long0 = frequency0.getCount((Object) integer0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCount(3181);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct((-1L));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct('Y');
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct(1073741824);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(31).when(comparator0).compare(any() , any());
      Frequency frequency0 = new Frequency(comparator0);
      Integer integer0 = new Integer(0);
      frequency0.addValue(integer0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      // Undeclared exception!
      try { 
        frequency0.addValue((Object) frequency0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // class (org.apache.commons.math.stat.Frequency) does not implement Comparable
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((Object) "C!V:aRT<(%h\"Oe,F");
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((Comparable<?>) 3181);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue(31);
      frequency0.addValue(31);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Comparator<String> comparator0 = (Comparator<String>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(0, 0, 0, 0, 0).when(comparator0).compare(anyString() , anyString());
      Frequency frequency0 = new Frequency(comparator0);
      frequency0.addValue((-1564L));
      String string0 = frequency0.toString();
      assertEquals("Value \t Freq. \t Pct. \t Cum Pct. \n-1564\t1\t100%\t100%\n", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue(5);
      long long0 = frequency0.getCumFreq(0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(31, 1, 1, (-826), 31).when(comparator0).compare(any() , any());
      Frequency frequency0 = new Frequency(comparator0);
      frequency0.addValue((-826));
      double double0 = frequency0.getCumPct('h');
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.hashCode();
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Frequency frequency1 = new Frequency();
      boolean boolean0 = frequency0.equals(frequency1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      boolean boolean0 = frequency0.equals(frequency0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      boolean boolean0 = frequency0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Object object0 = new Object();
      boolean boolean0 = frequency0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Frequency frequency1 = new Frequency();
      assertTrue(frequency1.equals((Object)frequency0));
      
      frequency1.addValue((-21));
      boolean boolean0 = frequency0.equals(frequency1);
      assertFalse(frequency1.equals((Object)frequency0));
      assertFalse(boolean0);
  }
}
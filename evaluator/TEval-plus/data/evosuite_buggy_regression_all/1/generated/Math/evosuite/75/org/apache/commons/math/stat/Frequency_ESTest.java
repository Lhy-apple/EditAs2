/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:07:40 GMT 2023
 */

package org.apache.commons.math.stat;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Comparator;
import java.util.Iterator;
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
      // Undeclared exception!
      try { 
        frequency0.getCumPct((Object) frequency0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.math.stat.Frequency cannot be cast to java.lang.Comparable
         //
         verifyException("org.apache.commons.math.stat.Frequency", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCumFreq(1717986918);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      // Undeclared exception!
      try { 
        frequency0.getCumFreq((Object) frequency0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.math.stat.Frequency cannot be cast to java.lang.Comparable
         //
         verifyException("org.apache.commons.math.stat.Frequency", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.clear();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer(709);
      frequency0.addValue(integer0);
      long long0 = frequency0.getCumFreq((Comparable<?>) integer0);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      // Undeclared exception!
      try { 
        frequency0.getPct((Object) frequency0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.math.stat.Frequency cannot be cast to java.lang.Comparable
         //
         verifyException("org.apache.commons.math.stat.Frequency", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct(0L);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCount('\'');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCumFreq('f');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct(35);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct('w');
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue('x');
      long long0 = frequency0.getCumFreq('f');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      // Undeclared exception!
      try { 
        frequency0.getCount((Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.TreeMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCount(2578);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct(1062L);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct('r');
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct(2152);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue(0L);
      double double0 = frequency0.getPct((Comparable<?>) 32);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
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
  public void test19()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer(769);
      frequency0.addValue((Object) integer0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((-671));
      frequency0.addValue((long) (-671));
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(0, 0, 0, 0, 0).when(comparator0).compare(any() , any());
      Frequency frequency0 = new Frequency(comparator0);
      frequency0.addValue(4823);
      String string0 = frequency0.toString();
      assertEquals("Value \t Freq. \t Pct. \t Cum Pct. \n4823\t1\t100%\t100%\n", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((-4640));
      Long long0 = new Long((-736L));
      double double0 = frequency0.getCumPct((Comparable<?>) long0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((-4640));
      frequency0.addValue((long) 2526);
      Long long0 = new Long((-736L));
      double double0 = frequency0.getCumPct((Comparable<?>) long0);
      assertEquals(0.5, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.hashCode();
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Frequency frequency1 = new Frequency();
      boolean boolean0 = frequency1.equals(frequency0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      boolean boolean0 = frequency0.equals(frequency0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      boolean boolean0 = frequency0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Iterator<Comparable<?>> iterator0 = frequency0.valuesIterator();
      boolean boolean0 = frequency0.equals(iterator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue((int) '(');
      Frequency frequency1 = new Frequency();
      boolean boolean0 = frequency1.equals(frequency0);
      assertFalse(boolean0);
  }
}

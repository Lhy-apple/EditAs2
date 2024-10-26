/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:10:33 GMT 2023
 */

package org.apache.commons.math.stat;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
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
      frequency0.addValue(2947);
      long long0 = frequency0.getCumFreq((-2704));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue(249L);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.clear();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct(1L);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCount('8');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Frequency frequency0 = new Frequency((Comparator) null);
      long long0 = frequency0.getCumFreq('u');
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct(258);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getCumPct('D');
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      frequency0.addValue('e');
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer((-2704));
      frequency0.addValue(integer0);
      frequency0.addValue((-1));
      long long0 = frequency0.getCumFreq((-547));
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      long long0 = frequency0.getCount((-2704));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Frequency frequency0 = new Frequency((Comparator) null);
      double double0 = frequency0.getCumPct(0L);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      double double0 = frequency0.getPct(':');
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Frequency frequency0 = new Frequency((Comparator) null);
      double double0 = frequency0.getCumPct((-2631));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer((-2704));
      frequency0.addValue(integer0);
      double double0 = frequency0.getPct((Object) integer0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer(0);
      frequency0.addValue(integer0);
      String string0 = frequency0.toString();
      assertEquals("Value \t Freq. \t Pct. \t Cum Pct. \n0\t1\t100%\t100%\n", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer(0);
      frequency0.addValue((Object) integer0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Frequency frequency0 = new Frequency();
      Integer integer0 = new Integer((-2704));
      frequency0.addValue(integer0);
      frequency0.addValue((-2704));
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Integer integer0 = new Integer((-2704));
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn((-2704), 1068, 1068, (-1), 1882).when(comparator0).compare(any() , any());
      Frequency frequency0 = new Frequency(comparator0);
      frequency0.addValue(2252);
      long long0 = frequency0.getCumFreq((Object) integer0);
      assertEquals(0L, long0);
  }
}

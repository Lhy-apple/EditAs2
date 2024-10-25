/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:04:47 GMT 2023
 */

package org.apache.commons.math3.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import org.apache.commons.math3.distribution.DiscreteDistribution;
import org.apache.commons.math3.util.Pair;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DiscreteDistribution_ESTest extends DiscreteDistribution_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      Long long0 = new Long(7L);
      Double double0 = new Double(7L);
      Pair<Long, Double> pair0 = new Pair<Long, Double>(long0, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
      discreteDistribution0.reseedRandomGenerator(7L);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      Long long0 = new Long((-10L));
      Double double0 = new Double((-10L));
      Pair<Long, Double> pair0 = new Pair<Long, Double>(long0, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = null;
      try {
        discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -10 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.distribution.DiscreteDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Double double0 = new Double(582L);
      Pair<Long, Double> pair0 = new Pair<Long, Double>((Long) null, double0);
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
      double double1 = discreteDistribution0.probability((Long) null);
      assertEquals(1.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      Double double0 = new Double(0.5215070554207777);
      Pair<String, Double> pair0 = new Pair<String, Double>("7TD7&T)", double0);
      linkedList0.add(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      double double1 = discreteDistribution0.probability("7TD7&T)");
      assertEquals(1.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      LinkedList<Pair<Object, Double>> linkedList0 = new LinkedList<Pair<Object, Double>>();
      Long long0 = new Long(1);
      Double double0 = Double.valueOf((double) 1);
      Pair<Object, Double> pair0 = new Pair<Object, Double>(long0, double0);
      linkedList0.offerLast(pair0);
      DiscreteDistribution<Object> discreteDistribution0 = new DiscreteDistribution<Object>(linkedList0);
      double double1 = discreteDistribution0.probability((Object) null);
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      Double double0 = new Double(0.5215070554207777);
      Pair<String, Double> pair0 = new Pair<String, Double>("7TD7&T)", double0);
      linkedList0.add(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      double double1 = discreteDistribution0.probability("EeObdx@");
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      Double double0 = new Double(387.20211956);
      Pair<Long, Double> pair0 = new Pair<Long, Double>((Long) null, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
      List<Pair<Long, Double>> list0 = discreteDistribution0.getSamples();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      Double double0 = new Double(387.20211956);
      Pair<Long, Double> pair0 = new Pair<Long, Double>((Long) null, double0);
      linkedList0.add(pair0);
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
      Long long0 = discreteDistribution0.sample();
      assertNull(long0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      Long long0 = new Long(0L);
      Double double0 = new Double(3.517957888583024);
      Pair<Long, Double> pair0 = new Pair<Long, Double>(long0, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
      Long[] longArray0 = discreteDistribution0.sample(577);
      assertEquals(577, longArray0.length);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      LinkedList<Pair<Long, Double>> linkedList0 = new LinkedList<Pair<Long, Double>>();
      Double double0 = new Double(387.20211956);
      Pair<Long, Double> pair0 = new Pair<Long, Double>((Long) null, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Long> discreteDistribution0 = new DiscreteDistribution<Long>(linkedList0);
      try { 
        discreteDistribution0.sample(0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // number of samples (0)
         //
         verifyException("org.apache.commons.math3.distribution.DiscreteDistribution", e);
      }
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:27:03 GMT 2023
 */

package org.apache.commons.math3.random;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.random.ISAACRandom;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.Well19937a;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.random.Well44497b;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BitsStreamGenerator_ESTest extends BitsStreamGenerator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Well44497b well44497b0 = new Well44497b();
      long long0 = well44497b0.nextLong();
      assertEquals((-3309472111078198868L), long0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Well44497b well44497b0 = new Well44497b(1826);
      int int0 = well44497b0.nextInt();
      assertEquals(1928307509, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      MersenneTwister mersenneTwister0 = new MersenneTwister(0);
      float float0 = mersenneTwister0.nextFloat();
      assertEquals(0.54881346F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Well19937c well19937c0 = new Well19937c(1826L);
      boolean boolean0 = well19937c0.nextBoolean();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      int[] intArray0 = new int[4];
      ISAACRandom iSAACRandom0 = new ISAACRandom(intArray0);
      boolean boolean0 = iSAACRandom0.nextBoolean();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Well19937a well19937a0 = new Well19937a();
      byte[] byteArray0 = new byte[5];
      well19937a0.nextBytes(byteArray0);
      assertArrayEquals(new byte[] {(byte) (-110), (byte) (-65), (byte)92, (byte) (-40), (byte) (-122)}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Well44497b well44497b0 = new Well44497b(776);
      double double0 = well44497b0.nextGaussian();
      assertEquals(1.619031634176317, double0, 0.01);
      
      double double1 = well44497b0.nextGaussian();
      assertEquals(0.3906881785797748, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Well19937a well19937a0 = new Well19937a();
      try { 
        well19937a0.nextInt((int) (byte) (-108));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -108 is smaller than, or equal to, the minimum (0)
         //
         verifyException("org.apache.commons.math3.random.BitsStreamGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Well44497b well44497b0 = new Well44497b((long) (byte) (-128));
      int int0 = well44497b0.nextInt(32);
      assertEquals(11, int0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Well19937c well19937c0 = new Well19937c(1826L);
      Well19937a well19937a0 = new Well19937a();
      Well44497b well44497b0 = new Well44497b();
      int int0 = well44497b0.nextInt(1857174117);
      assertEquals(1345755985, int0);
  }
}
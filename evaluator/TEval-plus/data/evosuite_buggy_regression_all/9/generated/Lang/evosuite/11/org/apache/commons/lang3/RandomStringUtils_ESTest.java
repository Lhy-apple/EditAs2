/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:24:40 GMT 2023
 */

package org.apache.commons.lang3;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Random;
import org.apache.commons.lang3.RandomStringUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockRandom;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RandomStringUtils_ESTest extends RandomStringUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String string0 = RandomStringUtils.randomAscii(26);
      assertEquals("                          ", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      char[] charArray0 = new char[1];
      // Undeclared exception!
      try { 
        RandomStringUtils.random(56320, 56320, 56320, true, true, charArray0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomAlphabetic(1934);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String string0 = RandomStringUtils.random(1934);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        RandomStringUtils.randomNumeric((-3281));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requested random string length -3281 is less than 0.
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      RandomStringUtils randomStringUtils0 = new RandomStringUtils();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      char[] charArray0 = new char[0];
      MockRandom mockRandom0 = new MockRandom();
      // Undeclared exception!
      try { 
        RandomStringUtils.random(1934, 29, (-1), true, false, charArray0, (Random) mockRandom0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The chars array must not be empty
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.random(56190, "i3oaO&j3Or");
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      char[] charArray0 = new char[6];
      MockRandom mockRandom0 = new MockRandom(0L);
      // Undeclared exception!
      RandomStringUtils.random(22, 0, 0, true, true, charArray0, (Random) mockRandom0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomNumeric(1934);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      org.evosuite.runtime.Random.setNextRandom(524);
      // Undeclared exception!
      RandomStringUtils.randomAlphanumeric(46340);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      org.evosuite.runtime.Random.setNextRandom(1934);
      String string0 = RandomStringUtils.randomAlphanumeric(956);
      assertEquals("77777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      org.evosuite.runtime.Random.setNextRandom((-1450));
      MockRandom mockRandom0 = new MockRandom(0L);
      // Undeclared exception!
      RandomStringUtils.random(56191, 55296, (-2293), false, false, (char[]) null, (Random) mockRandom0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      RandomStringUtils.random(4, (-1), 4, false, false);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      RandomStringUtils.random(26, 56191, 26, false, false);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.random(29, 56191, 29, false, false);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.random(56319, 56192, 317, false, false);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.random(56178, (String) null);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      RandomStringUtils.random(0, (char[]) null);
  }
}

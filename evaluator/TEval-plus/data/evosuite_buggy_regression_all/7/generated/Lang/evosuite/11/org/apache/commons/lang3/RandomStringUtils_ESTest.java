/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:36:20 GMT 2023
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
      String string0 = RandomStringUtils.randomAscii(1077);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      char[] charArray0 = new char[0];
      // Undeclared exception!
      try { 
        RandomStringUtils.random(1056, 1056, 1102, false, false, charArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The chars array must not be empty
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomAlphabetic(1115);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomAlphanumeric(430);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String string0 = RandomStringUtils.random(1931);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        RandomStringUtils.randomNumeric((-880));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requested random string length -880 is less than 0.
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      RandomStringUtils randomStringUtils0 = new RandomStringUtils();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String string0 = RandomStringUtils.random(2494, "}`~x(YH?,");
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      char[] charArray0 = new char[1];
      String string0 = RandomStringUtils.random(430, 0, 0, false, false, charArray0, (Random) mockRandom0);
      assertEquals("\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomNumeric(56319);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      org.evosuite.runtime.Random.setNextRandom(430);
      String string0 = RandomStringUtils.randomAlphanumeric(430);
      assertEquals("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      org.evosuite.runtime.Random.setNextRandom(430);
      MockRandom mockRandom0 = new MockRandom();
      String string0 = RandomStringUtils.random(128, 27, 430, false, true, (char[]) null, (Random) mockRandom0);
      assertEquals("66666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String string0 = RandomStringUtils.random(1540, (-880), 1540, true, false);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      // Undeclared exception!
      RandomStringUtils.random(519, 56334, 791, false, false, (char[]) null, (Random) mockRandom0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      // Undeclared exception!
      RandomStringUtils.random(123, 55296, (-1403), false, false, (char[]) null, (Random) mockRandom0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom(56192);
      // Undeclared exception!
      RandomStringUtils.random(56192, 56192, 37, false, false, (char[]) null, (Random) mockRandom0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = RandomStringUtils.random(0, (String) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String string0 = RandomStringUtils.random(0, (char[]) null);
      assertEquals("", string0);
  }
}
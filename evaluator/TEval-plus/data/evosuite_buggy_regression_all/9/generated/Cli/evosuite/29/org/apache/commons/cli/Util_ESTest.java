/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:36:32 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.cli.Util;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Util_ESTest extends Util_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Util util0 = new Util();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      String string0 = Util.stripLeadingHyphens("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = Util.stripLeadingHyphens((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      String string0 = Util.stripLeadingHyphens("--");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Util.stripLeadingHyphens("-");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Util.stripLeadingAndTrailingQuotes("\"]`3/XCiFMt");
      assertEquals("]`3/XCiFMt", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = Util.stripLeadingAndTrailingQuotes("k3UDO6\"");
      assertEquals("k3UDO6", string0);
  }
}
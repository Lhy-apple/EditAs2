/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:43:42 GMT 2023
 */

package org.apache.commons.lang3;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.File;
import org.apache.commons.lang3.SystemUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SystemUtils_ESTest extends SystemUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat("9");
      assertEquals(9.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      int[] intArray0 = SystemUtils.toJavaVersionIntArray("_WYFawQ{eCj:[");
      assertEquals(0, intArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionInt("mixed mode");
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = SystemUtils.getJavaIoTmpDir();
      assertTrue(file0.isAbsolute());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = SystemUtils.getJavaHome();
      assertFalse(file0.exists());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = SystemUtils.getUserDir();
      assertEquals("TEval-plus", file0.getName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      File file0 = SystemUtils.getUserHome();
      assertEquals("/home", file0.getParent());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SystemUtils systemUtils0 = new SystemUtils();
      assertFalse(SystemUtils.IS_OS_WINDOWS_98);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaAwtHeadless();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(27.614605F);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast((-1503.879F));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(Integer.MAX_VALUE);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionMatch("G", "G");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionMatch((String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch((String) null, (String) null, (String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("mixed mode", "Java Virtual Machine Specification", "lib", "\n");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("Dhg}i", (String) null, (String) null, "Dhg}i");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("Array element ", "Array element ", "Array element ", "Array element ");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("user.timezone", "wp08.|/|ql-[^hW9(NWf4", "user.timezone", "user.timezone");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSNameMatch("LU\u0006nBJXf", "LU\u0006nBJXf");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSNameMatch((String) null, "Dhg}i");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat((String) null);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat("1.8.0_202-b08");
      assertEquals(1.8F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat("amd64");
      assertEquals(64.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionInt("sun.awt.X11.XToolkit");
      assertEquals(1100.0F, float0, 0.01F);
  }
}
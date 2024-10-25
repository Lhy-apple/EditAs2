/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:02:28 GMT 2023
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
      float float0 = SystemUtils.toJavaVersionFloat("");
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      int[] intArray0 = SystemUtils.toJavaVersionIntArray("WO~luK");
      assertEquals(0, intArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionInt((String) null);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = SystemUtils.getJavaIoTmpDir();
      assertTrue(file0.isDirectory());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = SystemUtils.getJavaHome();
      assertFalse(file0.canRead());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = SystemUtils.getUserDir();
      assertEquals(0L, file0.getFreeSpace());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      File file0 = SystemUtils.getUserHome();
      assertTrue(file0.isAbsolute());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SystemUtils systemUtils0 = new SystemUtils();
      assertTrue(SystemUtils.IS_OS_LINUX);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaAwtHeadless();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(1450.3417F);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(0.0F);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(3900);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(2);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionMatch("Mac", "Mac");
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
      boolean boolean0 = SystemUtils.isOSMatch("'U8kD", "'U8kD", "'U8kD", "'U8kD");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("Cannot store ", (String) null, (String) null, "Cannot store ");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("'U8kD", "'U8kD", "|zM/jf^7wv(gWUb=3n", "|zM/jf^7wv(gWUb=3n");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("9b0b~*oXn", "9b0b~*oXn", "9b0b~*oXn", "/data/lhy/");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSNameMatch("J}", "J}");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSNameMatch((String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat("25.202-b08/tmp/EvoSuite_pathingJar983363495075357171.jar");
      assertEquals(25.2028F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat("SunOSLinux");
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionInt("'U8kD");
      assertEquals(800.0F, float0, 0.01F);
  }
}

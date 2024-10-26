/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:23:33 GMT 2023
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
      int[] intArray0 = SystemUtils.toJavaVersionIntArray("AIX");
      assertEquals(0, intArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      File file0 = SystemUtils.getJavaIoTmpDir();
      assertEquals("/tmp", file0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = SystemUtils.getJavaHome();
      assertEquals(0L, file0.getFreeSpace());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = SystemUtils.getUserDir();
      assertEquals("/data/lhy", file0.getParent());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = SystemUtils.getUserHome();
      assertEquals(0L, file0.getFreeSpace());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SystemUtils systemUtils0 = new SystemUtils();
      assertFalse(SystemUtils.IS_OS_WINDOWS_95);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaAwtHeadless();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(1136.0F);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(0.0F);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast(1038);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionAtLeast((-37));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionMatch(",m{'7~{]enQlcb", ",m{'7~{]enQlcb");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = SystemUtils.isJavaVersionMatch((String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch((String) null, (String) null, (String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("4;hLois>B2I6}", (String) null, (String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("tz4{4/z:mc", "tz4{4/z:mc", "tz4{4/z:mc", "tz4{4/z:mc");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("m8J*m4{pd", "m8J*m4{pd", "1.3", "m8J*m4{pd");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSMatch("", "", "", "rlb)*]z");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSNameMatch(",m{'7~{]enQlcb", ",m{'7~{]enQlcb");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      boolean boolean0 = SystemUtils.isOSNameMatch((String) null, (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat((String) null);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionInt("America/Los_Angeles1.8.0_202");
      assertEquals(180.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionFloat("5.0");
      assertEquals(5.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      float float0 = SystemUtils.toJavaVersionInt("java.runtime.version");
      assertEquals(0.0F, float0, 0.01F);
  }
}

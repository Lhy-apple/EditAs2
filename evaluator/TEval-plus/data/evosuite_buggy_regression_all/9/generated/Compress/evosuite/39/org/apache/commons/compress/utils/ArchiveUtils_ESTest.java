/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:26:08 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.utils.ArchiveUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArchiveUtils_ESTest extends ArchiveUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("org.apache.commonscompress.archivers.zip.ZipS[ort");
      String string0 = ArchiveUtils.toAsciiString(byteArray0, 0, 0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        ArchiveUtils.isEqual((byte[]) null, 4, 4, (byte[]) null, 4, 4);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      boolean boolean0 = ArchiveUtils.matchAsciiBuffer("", byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      boolean boolean0 = ArchiveUtils.isEqualWithNull((byte[]) null, 0, 0, (byte[]) null, 0, 0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("");
      byte[] byteArray1 = new byte[1];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      String string0 = ArchiveUtils.toAsciiString(byteArray0);
      assertEquals("\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("a[R%");
      String string0 = ArchiveUtils.toString((ArchiveEntry) jarArchiveEntry0);
      assertEquals("-      -1 a[R%", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "");
      String string0 = ArchiveUtils.toString((ArchiveEntry) tarArchiveEntry0);
      assertEquals("d       0 /", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("t");
      byte[] byteArray1 = new byte[1];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("");
      byte[] byteArray1 = new byte[1];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray1, byteArray0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("");
      byte[] byteArray1 = new byte[1];
      byteArray1[0] = (byte)50;
      boolean boolean0 = ArchiveUtils.isEqual(byteArray1, byteArray0, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("");
      byte[] byteArray1 = new byte[1];
      byteArray1[0] = (byte)50;
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = ArchiveUtils.isArrayZero((byte[]) null, 0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[1] = (byte)16;
      boolean boolean0 = ArchiveUtils.isArrayZero(byteArray0, 37);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String string0 = ArchiveUtils.sanitize("G|HW0`Y}5''@)r>");
      assertEquals("G|HW0`Y}?5''@)r>", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = ArchiveUtils.sanitize("\u0000\u0000\u0000\uFFFD\u0000");
      assertEquals("?????", string0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:48:32 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
import java.util.zip.ZipEntry;
import org.apache.commons.compress.archivers.zip.AsiExtraField;
import org.apache.commons.compress.archivers.zip.JarMarker;
import org.apache.commons.compress.archivers.zip.UnicodeCommentExtraField;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipShort;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveEntry_ESTest extends ZipArchiveEntry_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = null;
      try {
        zipArchiveEntry1 = new ZipArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ZIP compression method can not be negative: -1
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getLastModifiedDate();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockFile mockFile0 = new MockFile(")(w7e>3l%7!", ")(w7e>3l%7!");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, ")(w7e>3l%7!");
      zipArchiveEntry0.hashCode();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N");
      byte[] byteArray0 = zipArchiveEntry0.getCentralDirectoryExtra();
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N");
      zipArchiveEntry0.setPlatform((-1565));
      assertEquals((-1565), zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      int int0 = zipArchiveEntry0.getPlatform();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      zipArchiveEntry0.setMethod(3);
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry((ZipEntry) zipArchiveEntry0);
      ZipArchiveEntry zipArchiveEntry2 = new ZipArchiveEntry(zipArchiveEntry1);
      assertEquals(3, zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry2.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "org.apache.commons.compress.archivers.zip.UnicodeCommentExtraField/");
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertTrue(boolean0);
      assertNotSame(zipArchiveEntry1, zipArchiveEntry0);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[21];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertFalse(zipArchiveEntry1.isSupportedCompressionMethod());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setMethod(0);
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(0, zipArchiveEntry0.getMethod());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N");
      zipArchiveEntry0.setMethod(8);
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      zipArchiveEntry0.setUnixMode((-1655));
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7>/");
      zipArchiveEntry0.setUnixMode(25461);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("ZIP compression method can not be negative: ");
      zipArchiveEntry0.setUnixMode(0);
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(3, zipArchiveEntry0.getPlatform());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MockFile mockFile0 = new MockFile(")(w7e>3l%7!", ")(w7e>3l%7!");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, ")(w7e>3l%7!");
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(0, int0);
      assertEquals(0L, zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      UnicodeCommentExtraField unicodeCommentExtraField0 = new UnicodeCommentExtraField();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.addAsFirstExtraField(unicodeCommentExtraField0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.AbstractUnicodeExtraField", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      JarMarker jarMarker0 = new JarMarker();
      zipArchiveEntry0.addAsFirstExtraField(jarMarker0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      JarMarker jarMarker0 = new JarMarker();
      ZipShort zipShort0 = jarMarker0.getHeaderId();
      byte[] byteArray0 = new byte[3];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField(zipShort0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField((ZipShort) null);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      JarMarker jarMarker0 = new JarMarker();
      ZipShort zipShort0 = jarMarker0.getHeaderId();
      zipArchiveEntry0.addExtraField(jarMarker0);
      zipArchiveEntry0.removeExtraField(zipShort0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      AsiExtraField asiExtraField0 = new AsiExtraField();
      ZipShort zipShort0 = asiExtraField0.getLocalFileDataLength();
      zipArchiveEntry0.getExtraField(zipShort0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[21];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      MockFile mockFile0 = new MockFile(")(w7e>3l%7!", ")(w7e>3l%7!");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, ")(w7e>3l%7!");
      byte[] byteArray0 = zipArchiveEntry0.getLocalFileDataExtra();
      assertEquals(0, byteArray0.length);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertNotNull(byteArray0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[3];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.getLocalFileDataExtra();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[4];
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      zipArchiveEntry1.setCentralDirectoryExtra(byteArray0);
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertFalse(zipArchiveEntry1.isSupportedCompressionMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[21];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertTrue(boolean0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      boolean boolean0 = zipArchiveEntry0.equals((Object) null);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      AsiExtraField asiExtraField0 = new AsiExtraField();
      boolean boolean0 = zipArchiveEntry0.equals(asiExtraField0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertFalse(boolean0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      zipArchiveEntry0.setMethod(1);
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry(zipArchiveEntry0);
      zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(1, zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      assertEquals((-1), zipArchiveEntry0.getMethod());
      
      zipArchiveEntry0.setMethod(1);
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry(zipArchiveEntry0);
      zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(1, zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(")jK");
      zipArchiveEntry0.setMethod(9);
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry(zipArchiveEntry0);
      ZipArchiveEntry zipArchiveEntry2 = new ZipArchiveEntry((ZipEntry) zipArchiveEntry0);
      zipArchiveEntry1.equals(zipArchiveEntry2);
      assertEquals(9, zipArchiveEntry0.getMethod());
  }
}

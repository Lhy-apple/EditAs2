/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:22:27 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
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
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getCentralDirectoryExtra();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.hashCode();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      ZipArchiveEntry zipArchiveEntry2 = null;
      try {
        zipArchiveEntry2 = new ZipArchiveEntry(zipArchiveEntry1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ZIP compression method can not be negative: -1
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[7];
      zipArchiveEntry0.setExtra(byteArray0);
      AsiExtraField asiExtraField0 = new AsiExtraField();
      ZipShort zipShort0 = asiExtraField0.getHeaderId();
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
  public void test06()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[7];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setExtra(byteArray0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setPlatform(1511);
      assertEquals(1511, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      int int0 = zipArchiveEntry0.getPlatform();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, int0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile(",", ",");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, ",");
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFile mockFile0 = new MockFile("/");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "/");
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      ZipArchiveEntry zipArchiveEntry2 = (ZipArchiveEntry)zipArchiveEntry1.clone();
      assertEquals(0, zipArchiveEntry2.getInternalAttributes());
      assertEquals(0, zipArchiveEntry2.getPlatform());
      assertEquals(0L, zipArchiveEntry2.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry2.getMethod());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setMethod(0);
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(0, zipArchiveEntry0.getMethod());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setMethod(8);
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setUnixMode(1764);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("/");
      zipArchiveEntry0.setUnixMode(65280);
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals((-16777199L), zipArchiveEntry0.getExternalAttributes());
      assertEquals(65280, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, int0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      byte[] byteArray0 = new byte[7];
      zipArchiveEntry1.setExtra(byteArray0);
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      JarMarker jarMarker0 = new JarMarker();
      zipArchiveEntry0.addExtraField(jarMarker0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      JarMarker jarMarker0 = new JarMarker();
      zipArchiveEntry0.addAsFirstExtraField(jarMarker0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      JarMarker jarMarker0 = new JarMarker();
      zipArchiveEntry1.addAsFirstExtraField(jarMarker0);
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertNotSame(zipArchiveEntry1, zipArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      UnicodeCommentExtraField unicodeCommentExtraField0 = new UnicodeCommentExtraField();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField(unicodeCommentExtraField0.UCOM_ID);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[7];
      zipArchiveEntry0.setExtra(byteArray0);
      ZipShort zipShort0 = new ZipShort(byteArray0);
      zipArchiveEntry0.removeExtraField(zipShort0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("/");
      UnicodeCommentExtraField unicodeCommentExtraField0 = new UnicodeCommentExtraField();
      zipArchiveEntry0.getExtraField(unicodeCommentExtraField0.UCOM_ID);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = zipArchiveEntry0.getLocalFileDataExtra();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertNotNull(byteArray0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, byteArray0.length);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      zipArchiveEntry1.getLocalFileDataExtra();
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getPlatform());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setName("f;Hj,|V=6(R");
      boolean boolean0 = zipArchiveEntry0.isDirectory();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[7];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry0);
      assertTrue(boolean0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals((Object) null);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      Object object0 = new Object();
      boolean boolean0 = zipArchiveEntry0.equals(object0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertFalse(boolean0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      zipArchiveEntry1.setName("ZIP compression method can not be negative: ");
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setName("");
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry("");
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setName("bad CRC checksum ");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }
}

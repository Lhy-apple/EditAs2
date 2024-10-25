/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:05:58 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.util.NoSuchElementException;
import org.apache.commons.compress.archivers.zip.AsiExtraField;
import org.apache.commons.compress.archivers.zip.JarMarker;
import org.apache.commons.compress.archivers.zip.UnicodeCommentExtraField;
import org.apache.commons.compress.archivers.zip.UnicodePathExtraField;
import org.apache.commons.compress.archivers.zip.UnrecognizedExtraField;
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
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("8[");
      zipArchiveEntry0.setUnixMode(128);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
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
  public void test02()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      zipArchiveEntry0.getLastModifiedDate();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      zipArchiveEntry0.hashCode();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
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
  public void test06()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("9yBDMd1sx5g");
      zipArchiveEntry0.setPlatform(3);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      int int0 = zipArchiveEntry0.getPlatform();
      assertEquals(0, int0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      File file0 = MockFile.createTempFile(" 4os", " 4os");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(file0, ".G|(Z");
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getSize());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "org.apache.commons.compress.archivers.zip.UnicodeCommentExtraField/");
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G");
      assertEquals((-1), zipArchiveEntry0.getMethod());
      
      zipArchiveEntry0.setMethod(0);
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(0, zipArchiveEntry0.getMethod());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      boolean boolean0 = zipArchiveEntry0.isSupportedCompressionMethod();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertFalse(boolean0);
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
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      zipArchiveEntry0.setUnixMode(0);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("");
      zipArchiveEntry0.setUnixMode(0);
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(3, zipArchiveEntry0.getPlatform());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, int0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[4];
      byte[] byteArray1 = zipArchiveEntry0.getCentralDirectoryExtra();
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray1);
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N/");
      UnicodeCommentExtraField unicodeCommentExtraField0 = new UnicodeCommentExtraField();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.addExtraField(unicodeCommentExtraField0);
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
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      byte[] byteArray0 = new byte[4];
      UnicodeCommentExtraField unicodeCommentExtraField0 = new UnicodeCommentExtraField("o!.<6qLRG[m04!*I(*", byteArray0);
      zipArchiveEntry0.addAsFirstExtraField(unicodeCommentExtraField0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      UnrecognizedExtraField unrecognizedExtraField0 = new UnrecognizedExtraField();
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      // Undeclared exception!
      try { 
        zipArchiveEntry0.addAsFirstExtraField(unrecognizedExtraField0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.UnrecognizedExtraField", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N");
      byte[] byteArray0 = zipArchiveEntry0.getCentralDirectoryExtra();
      zipArchiveEntry0.setExtra(byteArray0);
      UnicodePathExtraField unicodePathExtraField0 = new UnicodePathExtraField();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField(unicodePathExtraField0.UPATH_ID);
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
      UnicodePathExtraField unicodePathExtraField0 = new UnicodePathExtraField();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("UslQ_");
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField(unicodePathExtraField0.UPATH_ID);
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
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      JarMarker jarMarker0 = JarMarker.getInstance();
      ZipShort zipShort0 = jarMarker0.getCentralDirectoryLength();
      zipArchiveEntry0.removeExtraField(zipShort0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      AsiExtraField asiExtraField0 = new AsiExtraField();
      ZipShort zipShort0 = asiExtraField0.getLocalFileDataLength();
      zipArchiveEntry0.getExtraField(zipShort0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = zipArchiveEntry0.getLocalFileDataExtra();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, byteArray0.length);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertNotNull(byteArray0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      byte[] byteArray1 = zipArchiveEntry0.getLocalFileDataExtra();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(4, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N/");
      zipArchiveEntry0.setName("ZIP compression method can not be negative: ");
      zipArchiveEntry0.getName();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("G@lM@5X7$");
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertTrue(boolean0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      boolean boolean0 = zipArchiveEntry0.equals((Object) null);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("na5>sf'H+<N");
      UnicodePathExtraField unicodePathExtraField0 = new UnicodePathExtraField();
      boolean boolean0 = zipArchiveEntry0.equals(unicodePathExtraField0.UPATH_ID);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      Object object0 = zipArchiveEntry0.clone();
      zipArchiveEntry0.setName("=_DFf;NP? 5+][e}5");
      boolean boolean0 = zipArchiveEntry0.equals(object0);
      assertFalse(object0.equals((Object)zipArchiveEntry0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      Object object0 = zipArchiveEntry0.clone();
      zipArchiveEntry0.setName("");
      boolean boolean0 = object0.equals(zipArchiveEntry0);
      assertFalse(object0.equals((Object)zipArchiveEntry0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("=_DFf;NP? 5+][e}5");
      zipArchiveEntry0.setName("=_DFf;NP? 5+][e}5");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertNotSame(zipArchiveEntry1, zipArchiveEntry0);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertTrue(boolean0);
  }
}

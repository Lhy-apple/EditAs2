/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:50:03 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.IllegalCharsetNameException;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveOutputStream_ESTest extends ZipArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockFile mockFile0 = new MockFile("-_Pr3|UF@?++.c8{{;", "=SMHt'r^");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      String string0 = zipArchiveOutputStream0.getEncoding();
      assertEquals("UTF8", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.setComment("");
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockFile mockFile0 = new MockFile("crc checksum is required for STORED method when not writing to a file", "z z\"o08Ozk2");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      zipArchiveOutputStream0.setMethod(0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("z z\"o08Ozk2");
      try { 
        zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Error in writing to file
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setFallbackToUTF8(true);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setEncoding("");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("FATAL: UTF-8 encoding not supported.", "FATAL: UTF-8 encoding not supported.");
      ArchiveEntry archiveEntry0 = zipArchiveOutputStream0.createArchiveEntry(mockFile0, "FATAL: UTF-8 encoding not supported.");
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertEquals("FATAL: UTF-8 encoding not supported.", archiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.ALWAYS;
      String string0 = zipArchiveOutputStream_UnicodeExtraFieldPolicy0.toString();
      assertEquals("always", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("never");
      File file0 = MockFile.createTempFile("never", "never", (File) mockFile0);
      MockFile mockFile1 = new MockFile(file0, "not encodeable");
      ZipArchiveOutputStream zipArchiveOutputStream0 = null;
      try {
        zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile1);
        fail("Expecting exception: FileNotFoundException");
      
      } catch(Throwable e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("", "");
      ZipArchiveOutputStream zipArchiveOutputStream0 = null;
      try {
        zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
        fail("Expecting exception: FileNotFoundException");
      
      } catch(Throwable e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      boolean boolean0 = zipArchiveOutputStream0.isSeekable();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      File file0 = MockFile.createTempFile(".>JOF9hT\"D:a2", "", (File) null);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(file0);
      boolean boolean0 = zipArchiveOutputStream0.isSeekable();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(false);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\u0000\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(true);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream0.closeArchiveEntry();
      zipArchiveOutputStream0.finish();
      assertEquals(116, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0003\u0000PK\u0007\b\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000\u0000\u0000\u0000\u0000PK\u0001\u0002\u0014\u0000\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000PK\u0005\u0006\u0000\u0000\u0000\u0000\u0001\u0000\u0001\u0000.\u0000\u0000\u00000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(2L, zipArchiveEntry0.getCompressedSize());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0003\u0000PK\u0007\b\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000\u0000\u0000\u0000\u0000PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setMethod(0);
      MockFile mockFile0 = new MockFile("always");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "always");
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // crc checksum is required for STORED method when not writing to a file
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setMethod(0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // uncompressed size is required for STORED method when not writing to a file
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.setLevel(0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(30, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setLevel((-2028178998));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid compression level: -2028178998
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setLevel(445);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid compression level: 445
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setLevel((-1));
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      ZipArchiveOutputStream zipArchiveOutputStream1 = new ZipArchiveOutputStream(zipArchiveOutputStream0);
      try { 
        zipArchiveOutputStream1.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = zipArchiveEntry0.getCentralDirectoryExtra();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.write(byteArray0, 9578, 9578);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.zip.Deflater", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      zipArchiveOutputStream0.flush();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.flush();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.ALWAYS;
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(39, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\t\u0000up\u0005\u0000\u0001\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("zn}:H");
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields((ZipArchiveOutputStream.UnicodeExtraFieldPolicy) null);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("");
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.ALWAYS;
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("crc checksum is required for STORED method when not writing to a file");
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000W\u0000up\u0005\u0000\u0001\u0000\u0000\u0000\u0000ucJ\u0000\u0001#}\uFFFD\uFFFDcrc checksum is required for STORED method when not writing to a file", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.writeLocalFileHeader(zipArchiveEntry0);
      assertEquals(30, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\n\u0000\u0000\b\uFFFD\uFFFD\u0000!\u0000\u0000\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.writeDataDescriptor(zipArchiveEntry0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("Cannot determine type of file ");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.writeCentralFileHeader(zipArchiveEntry0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      MockFile mockFile0 = new MockFile("pA|Ri2'h ");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      try { 
        zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Error in writing to file
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }
}

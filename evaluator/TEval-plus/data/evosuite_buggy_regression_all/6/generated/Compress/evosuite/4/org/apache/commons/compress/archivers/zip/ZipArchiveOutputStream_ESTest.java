/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:11:12 GMT 2023
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
import java.util.jar.JarEntry;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveOutputStream_ESTest extends ZipArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP850");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      String string0 = zipArchiveOutputStream0.getEncoding();
      assertEquals("UTF8", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      JarEntry jarEntry0 = new JarEntry("ibm437");
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(jarEntry0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      jarArchiveEntry0.setMethod(0);
      try { 
        zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // bad CRC checksum for entry ibm437: ffffffffffffffff instead of 0
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setMethod(20);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setFallbackToUTF8(false);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setEncoding("S\"Qyy^%Q?bs");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // S\"Qyy^%Q?bs
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("i.", "CP50");
      ArchiveEntry archiveEntry0 = zipArchiveOutputStream0.createArchiveEntry(mockFile0, "Xv");
      zipArchiveOutputStream0.putArchiveEntry(archiveEntry0);
      zipArchiveOutputStream0.closeArchiveEntry();
      zipArchiveOutputStream0.finish();
      assertEquals(120, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\u0000!\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000Xv\u0003\u0000PK\u0007\b\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000\u0000\u0000\u0000\u0000PK\u0001\u0002\u0014\u0000\u0014\u0000\b\b\b\u0000\u0000!\u0000\u0000\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000XvPK\u0005\u0006\u0000\u0000\u0000\u0000\u0001\u0000\u0001\u00000\u0000\u0000\u00002\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      String string0 = zipArchiveOutputStream_UnicodeExtraFieldPolicy0.toString();
      assertEquals("not encodeable", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      File file0 = MockFile.createTempFile("x`D2L+YS", "rI{\\R[9%f=A[]{_9@=");
      MockFile mockFile0 = new MockFile(file0, "rI{\\R[9%f=A[]{_9@=");
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
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP850", "");
      FileSystemHandling.shouldAllThrowIOExceptions();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      boolean boolean0 = zipArchiveOutputStream0.isSeekable();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP850", "");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      boolean boolean0 = zipArchiveOutputStream0.isSeekable();
      assertTrue(boolean0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(false);
      MockFile mockFile0 = new MockFile("i.", "CP50");
      ZipArchiveEntry zipArchiveEntry0 = (ZipArchiveEntry)zipArchiveOutputStream0.createArchiveEntry(mockFile0, "Xv");
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\u0000\b\u0000\u0000!\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0002\u0000\u0000\u0000Xv", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setUseLanguageEncodingFlag(true);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("");
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      try { 
        zipArchiveOutputStream0.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockFile mockFile0 = new MockFile("d]v?");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "");
      JarEntry jarEntry0 = new JarEntry(zipArchiveEntry0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(jarEntry0);
      jarArchiveEntry0.setMethod(0);
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
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
  public void test14()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(zipArchiveEntry0);
      jarArchiveEntry0.setMethod(0);
      try { 
        zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // uncompressed size is required for STORED method when not writing to a file
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("}X");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFileOutputStream0);
      MockFile mockFile0 = new MockFile("}X");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "}X");
      JarEntry jarEntry0 = new JarEntry(zipArchiveEntry0);
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(jarEntry0);
      jarArchiveEntry0.setMethod(0);
      try { 
        zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
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
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.setLevel(3);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      assertEquals(8, zipArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setLevel((-2419));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid compression level: -2419
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.setLevel(948);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid compression level: 948
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP50");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      zipArchiveOutputStream0.setLevel((-1));
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      JarEntry jarEntry0 = new JarEntry("ibm437");
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry(jarEntry0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      byte[] byteArray0 = new byte[4];
      // Undeclared exception!
      try { 
        zipArchiveOutputStream0.write(byteArray0, 3, 16711688);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.zip.Deflater", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveOutputStream zipArchiveOutputStream1 = new ZipArchiveOutputStream(zipArchiveOutputStream0);
      zipArchiveOutputStream0.putArchiveEntry(zipArchiveEntry0);
      zipArchiveOutputStream1.putArchiveEntry(zipArchiveEntry0);
      assertEquals(30, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.close();
      assertEquals("PK\u0005\u0006\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
      assertEquals(22, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP850", "");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      zipArchiveOutputStream0.setComment("never");
      zipArchiveOutputStream0.close();
      assertEquals(27L, mockFile0.length());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP50");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      zipArchiveOutputStream0.flush();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.flush();
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.ALWAYS;
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("ZheIZf/(/2n]#");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(65, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\r\u0000\u0016\u0000ZheIZf/(/2n]#up\u0012\u0000\u0001w\uFFFD\uFFFD\uFFFDZheIZf/(/2n]#", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("]DL`sCE63ai\"k=X'{~");
      jarArchiveEntry0.setComment("]DL`sCE63ai\"k=X'{~");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(48, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0012\u0000\u0000\u0000]DL`sCE63ai\"k=X'{~", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.NOT_ENCODEABLE;
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("]DL`sCE63ai\"k=X'{~");
      jarArchiveEntry0.setComment("");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(48, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0012\u0000\u0000\u0000]DL`sCE63ai\"k=X'{~", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream.UnicodeExtraFieldPolicy zipArchiveOutputStream_UnicodeExtraFieldPolicy0 = ZipArchiveOutputStream.UnicodeExtraFieldPolicy.ALWAYS;
      JarArchiveEntry jarArchiveEntry0 = new JarArchiveEntry("ZheIZf/(/2n]#");
      jarArchiveEntry0.setComment("ZheIZf/(/2n]#");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      zipArchiveOutputStream0.setCreateUnicodeExtraFields(zipArchiveOutputStream_UnicodeExtraFieldPolicy0);
      zipArchiveOutputStream0.putArchiveEntry(jarArchiveEntry0);
      assertEquals(8, jarArchiveEntry0.getMethod());
      assertEquals("PK\u0003\u0004\u0014\u0000\b\b\b\u0000\uFFFD\uFFFDND\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\r\u0000,\u0000ZheIZf/(/2n]#up\u0012\u0000\u0001w\uFFFD\uFFFD\uFFFDZheIZf/(/2n]#uc\u0012\u0000\u0001w\uFFFD\uFFFD\uFFFDZheIZf/(/2n]#", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.writeLocalFileHeader(zipArchiveEntry0);
      assertEquals(30, byteArrayOutputStream0.size());
      assertEquals("PK\u0003\u0004\n\u0000\u0000\b\uFFFD\uFFFD\u0000!\u0000\u0000\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveOutputStream0.writeDataDescriptor(zipArchiveEntry0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(byteArrayOutputStream0);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment(": ");
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
  public void test33()  throws Throwable  {
      MockFile mockFile0 = new MockFile("CP850", "CP850");
      ZipArchiveOutputStream zipArchiveOutputStream0 = new ZipArchiveOutputStream(mockFile0);
      ArchiveEntry archiveEntry0 = zipArchiveOutputStream0.createArchiveEntry(mockFile0, "CP850");
      try { 
        zipArchiveOutputStream0.putArchiveEntry(archiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Error in writing to file
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }
}
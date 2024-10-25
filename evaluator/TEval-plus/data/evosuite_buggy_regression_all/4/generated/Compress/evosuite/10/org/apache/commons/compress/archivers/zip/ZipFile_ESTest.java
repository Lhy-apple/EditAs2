/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:29:30 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.IllegalCharsetNameException;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.zip.ZipFile;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipFile_ESTest extends ZipFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipFile$2", "m9)otZFZD7'Nb8OB");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(mockFile0, "");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("org.apache.commons.compress.archivers.zip.ZipFile", "org.apache.commons.compress.archivers.zip.ZipFile");
        fail("Expecting exception: FileNotFoundException");
      
      } catch(Throwable e) {
         //
         // File does not exist, and RandomAccessFile is not open in write mode
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockRandomAccessFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("");
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ZipFile.closeQuietly((ZipFile) null);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("/tmp/file.encoding0file.encoding");
      byte[] byteArray0 = new byte[37];
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      File file0 = MockFile.createTempFile("file.encoding", "file.encoding");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(file0);
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("/tmp/file.encoding0file.encoding");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, ",&PKx=<F");
      byte[] byteArray0 = new byte[37];
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      File file0 = MockFile.createTempFile("file.encoding", "file.encoding");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(file0);
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("/tmp/fble.encoding0fble.encoding");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, "archive's ZIP64 end of central directory locator is corrupt.");
      File file0 = MockFile.createTempFile("fble.encoding", "fble.encoding");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(file0);
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }
}

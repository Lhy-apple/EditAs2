/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:45:16 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.io.FileNotFoundException;
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
      MockFile mockFile0 = new MockFile("J.M", "central directory is empty, can't expand corrupt archive.");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(mockFile0, (String) null);
        fail("Expecting exception: FileNotFoundException");
      
      } catch(Throwable e) {
         //
         // File does not exist, and RandomAccessFile is not open in write mode
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockRandomAccessFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      File file0 = MockFile.createTempFile("UwFSIF8", "UwFSIF8");
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
  public void test2()  throws Throwable  {
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("This archives contains unclosed entries.", "This archives contains unclosed entries.");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ZipFile.closeQuietly((ZipFile) null);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("4dc");
      FileSystemHandling.appendLineToFile(evoSuiteFile0, "CR[PJ_Z");
      FileSystemHandling.appendLineToFile(evoSuiteFile0, "U8 )CuY\"/F)=2_wC");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("4dc");
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }
}
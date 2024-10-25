/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:42:24 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedReader;
import java.io.PipedWriter;
import java.io.StringReader;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.sql.ResultSetMetaData;
import java.util.Map;
import java.util.function.Consumer;
import javax.sql.rowset.RowSetMetaDataImpl;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVParser_ESTest extends CSVParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("");
      CSVParser cSVParser0 = cSVFormat0.parse(stringReader0);
      cSVParser0.getFirstEndOfLine();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      PipedReader pipedReader0 = new PipedReader(44);
      CSVParser cSVParser0 = cSVFormat0.parse(pipedReader0);
      cSVParser0.getCurrentLineNumber();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      MockFile mockFile0 = new MockFile("E/U!");
      Charset charset0 = Charset.defaultCharset();
      try { 
        CSVParser.parse((File) mockFile0, charset0, cSVFormat0);
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      URL uRL0 = MockURL.getFileExample();
      Charset charset0 = Charset.defaultCharset();
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      // Undeclared exception!
      try { 
        CSVParser.parse(uRL0, charset0, cSVFormat0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      Charset charset0 = Charset.defaultCharset();
      CSVParser cSVParser0 = CSVParser.parse((InputStream) pipedInputStream0, charset0, cSVFormat0);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      MockFile mockFile0 = new MockFile(") EOF reached before encapsulated token finished", "");
      Path path0 = mockFile0.toPath();
      Charset charset0 = Charset.defaultCharset();
      try { 
        CSVParser.parse(path0, charset0, cSVFormat0);
        fail("Expecting exception: NoSuchFileException");
      
      } catch(NoSuchFileException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVParser cSVParser0 = CSVParser.parse("IB", cSVFormat0);
      Consumer<Object> consumer0 = (Consumer<Object>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVParser0.forEach(consumer0);
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedReader pipedReader0 = new PipedReader(2);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVParser cSVParser0 = cSVFormat0.parse(pipedReader0);
      long long0 = cSVParser0.getRecordNumber();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withTrim();
      CSVParser cSVParser0 = CSVParser.parse("\u0085", cSVFormat1);
      cSVParser0.nextRecord();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.INFORMIX_UNLOAD_CSV;
      String string0 = "4Nny{/,CZj\\'{o\\uKK";
      CSVParser cSVParser0 = CSVParser.parse(string0, cSVFormat0);
      cSVParser0.nextRecord();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("charset");
      CSVParser cSVParser0 = CSVParser.parse("charset", cSVFormat1);
      cSVParser0.nextRecord();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[1];
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      CSVParser cSVParser0 = CSVParser.parse("\u0085", cSVFormat1);
      Map<String, Integer> map0 = cSVParser0.getHeaderMap();
      assertEquals(1, map0.size());
      assertEquals(0L, cSVParser0.getRecordNumber());
      assertNotNull(map0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      PipedReader pipedReader0 = new PipedReader();
      CSVParser cSVParser0 = new CSVParser(pipedReader0, cSVFormat0);
      cSVParser0.getHeaderMap();
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVParser cSVParser0 = CSVParser.parse("IB", cSVFormat0);
      cSVParser0.getRecords();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      CSVParser cSVParser0 = CSVParser.parse("", cSVFormat1);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      CSVParser cSVParser0 = CSVParser.parse("charset", cSVFormat1);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[1];
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      CSVFormat cSVFormat2 = cSVFormat1.withSkipHeaderRecord(true);
      CSVParser cSVParser0 = CSVParser.parse("path", cSVFormat2);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      String[] stringArray0 = new String[5];
      stringArray0[0] = "kC]zAVq(@t";
      stringArray0[1] = "reader";
      stringArray0[2] = "tU(5+";
      stringArray0[4] = "";
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      PipedWriter pipedWriter0 = new PipedWriter();
      PipedReader pipedReader0 = new PipedReader(pipedWriter0);
      CSVParser cSVParser0 = cSVFormat1.parse(pipedReader0);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.ORACLE;
      PipedReader pipedReader0 = new PipedReader(44);
      CSVParser cSVParser0 = cSVFormat0.parse(pipedReader0);
      Consumer<CSVRecord> consumer0 = (Consumer<CSVRecord>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVParser0.close();
      cSVParser0.forEach(consumer0);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }
}

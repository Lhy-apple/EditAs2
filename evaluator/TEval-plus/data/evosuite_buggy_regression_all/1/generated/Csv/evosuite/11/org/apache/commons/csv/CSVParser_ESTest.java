/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:27:51 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.io.IOException;
import java.io.PipedReader;
import java.io.StringReader;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Map;
import java.util.function.Consumer;
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
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVParser cSVParser0 = CSVParser.parse("lT", cSVFormat0);
      Consumer<CSVRecord> consumer0 = (Consumer<CSVRecord>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVParser0.forEach(consumer0);
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      URL uRL0 = MockURL.getFileExample();
      Charset charset0 = Charset.defaultCharset();
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
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVParser cSVParser0 = CSVParser.parse(") invalid parse sequence", cSVFormat0);
      long long0 = cSVParser0.getCurrentLineNumber();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile(" SkipHeadGrRecord:", ") EOF reached before encapsulated token finished", (File) null);
      Charset charset0 = Charset.defaultCharset();
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVParser cSVParser0 = CSVParser.parse(file0, charset0, cSVFormat0);
      long long0 = cSVParser0.getRecordNumber();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString(") invalid parse sequence");
      CSVParser cSVParser0 = CSVParser.parse(") invalid parse sequence", cSVFormat1);
      cSVParser0.nextRecord();
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString(") invaliB parse saqu-nce");
      CSVParser cSVParser0 = CSVParser.parse("w=bg7,\"n)u0x4", cSVFormat1);
      try { 
        cSVParser0.nextRecord();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (startline 1) EOF reached before encapsulated token finished
         //
         verifyException("org.apache.commons.csv.Lexer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[2];
      stringArray0[0] = "org.apache.commons.csv.csvparser$1";
      stringArray0[1] = "0sLWuc`ZgO|w";
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      StringReader stringReader0 = new StringReader("org.apache.commons.csv.csvparser$1");
      CSVParser cSVParser0 = new CSVParser(stringReader0, cSVFormat1);
      Map<String, Integer> map0 = cSVParser0.getHeaderMap();
      assertEquals(0L, cSVParser0.getRecordNumber());
      assertEquals(2, map0.size());
      assertNotNull(map0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVParser cSVParser0 = CSVParser.parse("The comment start character and the quoteChar cannot be the same ('", cSVFormat0);
      Map<String, Integer> map0 = cSVParser0.getHeaderMap();
      assertNull(map0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      String[] stringArray0 = new String[0];
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      StringReader stringReader0 = new StringReader("0sLWuc`ZgO|w");
      CSVParser cSVParser0 = new CSVParser(stringReader0, cSVFormat1);
      assertEquals(1L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      String[] stringArray0 = new String[0];
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      StringReader stringReader0 = new StringReader("0sLWuc`ZgO|w");
      CSVParser cSVParser0 = new CSVParser(stringReader0, cSVFormat0);
      cSVParser0.getRecords();
      assertEquals(1L, cSVParser0.getRecordNumber());
      
      CSVParser cSVParser1 = new CSVParser(stringReader0, cSVFormat1);
      assertEquals(0L, cSVParser1.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      String[] stringArray0 = new String[4];
      stringArray0[0] = "n)m)8C7jb<!/7R{a.";
      stringArray0[1] = "org.apache.commons.csv.CSVParser$1";
      stringArray0[2] = "o5\"(+K7;r";
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      CSVFormat cSVFormat2 = cSVFormat1.withSkipHeaderRecord(true);
      PipedReader pipedReader0 = new PipedReader();
      try { 
        cSVFormat2.parse(pipedReader0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedReader", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVParser cSVParser0 = CSVParser.parse("Delimiter=<", cSVFormat0);
      cSVParser0.close();
      Consumer<Object> consumer0 = (Consumer<Object>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVParser0.forEach(consumer0);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }
}
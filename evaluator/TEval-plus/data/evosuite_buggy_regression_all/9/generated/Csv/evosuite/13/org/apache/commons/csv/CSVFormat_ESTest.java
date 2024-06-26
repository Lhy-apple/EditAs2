/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:28:34 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.PipedReader;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import javax.sql.rowset.RowSetMetaDataImpl;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.QuoteMode;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFileWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVFormat_ESTest extends CSVFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[1];
      String string0 = cSVFormat0.TDF.format(objectArray0);
      assertEquals("\"\"", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      // Undeclared exception!
      try { 
        cSVFormat1.format((Object[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.csv.CSVPrinter", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      MockFileWriter mockFileWriter0 = new MockFileWriter("\"", true);
      CSVPrinter cSVPrinter0 = cSVFormat0.print(mockFileWriter0);
      assertNotNull(cSVPrinter0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withSkipHeaderRecord();
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CSVFormat.Predefined cSVFormat_Predefined0 = CSVFormat.Predefined.MySQL;
      CSVFormat cSVFormat0 = cSVFormat_Predefined0.getFormat();
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreSurroundingSpaces();
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('\u000E');
      CSVFormat cSVFormat1 = cSVFormat0.withRecordSeparator('\u000E');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat0.getSkipHeaderRecord());
      assertEquals('\u000E', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.isNullStringSet());
      assertEquals("\u000E", cSVFormat1.getRecordSeparator());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreHeaderCase();
      cSVFormat1.hashCode();
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
      assertTrue(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        CSVFormat.valueOf("The quoteChar character and the delimiter cannot be the same ('");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No enum constant org.apache.commons.csv.CSVFormat.Predefined.The quoteChar character and the delimiter cannot be the same ('
         //
         verifyException("java.lang.Enum", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[1];
      CSVFormat cSVFormat1 = cSVFormat0.TDF.withHeaderComments(objectArray0);
      MockFileWriter mockFileWriter0 = new MockFileWriter("\"", true);
      cSVFormat1.print(mockFileWriter0);
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isNullStringSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.MYSQL.withCommentMarker('`');
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      PipedReader pipedReader0 = new PipedReader();
      CSVParser cSVParser0 = cSVFormat0.TDF.parse(pipedReader0);
      assertFalse(cSVParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("");
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertEquals("", cSVFormat1.getNullString());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertEquals("\r\n", cSVFormat1.getRecordSeparator());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat0.getAllowMissingColumnNames();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      // Undeclared exception!
      try { 
        cSVFormat0.withEscape(',');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The escape character and the delimiter cannot be the same (',')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      // Undeclared exception!
      try { 
        cSVFormat0.withQuote('\n');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The quoteChar cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      // Undeclared exception!
      try { 
        cSVFormat0.withDelimiter('\r');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The delimiter cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape((Character) null);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Character character0 = Character.valueOf('#');
      Object[] objectArray0 = new Object[1];
      objectArray0[0] = (Object) character0;
      CSVFormat cSVFormat1 = cSVFormat0.TDF.withHeaderComments(objectArray0);
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape('\u000E');
      cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      boolean boolean0 = cSVFormat0.equals(cSVFormat0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      boolean boolean0 = cSVFormat0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('\u000E');
      boolean boolean0 = cSVFormat0.equals("Y");
      assertFalse(cSVFormat0.getSkipHeaderRecord());
      assertFalse(boolean0);
      assertEquals('\u000E', cSVFormat0.getDelimiter());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = CSVFormat.DEFAULT;
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      QuoteMode quoteMode0 = QuoteMode.NONE;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(boolean0);
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape('#');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertEquals('#', (char)cSVFormat1.getEscapeCharacter());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals('\"', (char)cSVFormat1.getQuoteCharacter());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = CSVFormat.MYSQL;
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('6');
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertEquals('6', (char)cSVFormat1.getCommentMarker());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker(';');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertEquals(';', (char)cSVFormat1.getCommentMarker());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('!');
      CSVFormat cSVFormat2 = cSVFormat1.withIgnoreSurroundingSpaces(false);
      boolean boolean0 = cSVFormat2.equals(cSVFormat1);
      assertFalse(cSVFormat2.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat2.isNullStringSet());
      assertEquals('!', (char)cSVFormat2.getCommentMarker());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(boolean0);
      assertFalse(cSVFormat2.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat2.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('\u000E');
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreHeaderCase();
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(boolean0);
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("");
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("");
      CSVFormat cSVFormat2 = cSVFormat1.withAllowMissingColumnNames();
      boolean boolean0 = cSVFormat1.equals(cSVFormat2);
      assertTrue(cSVFormat2.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals("", cSVFormat2.getNullString());
      assertTrue(cSVFormat2.getAllowMissingColumnNames());
      assertTrue(cSVFormat2.getIgnoreEmptyLines());
      assertTrue(boolean0);
      assertEquals("\r\n", cSVFormat2.getRecordSeparator());
      assertFalse(cSVFormat2.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreSurroundingSpaces(true);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = CSVFormat.RFC4180;
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withSkipHeaderRecord();
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withRecordSeparator('6');
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(boolean0);
      assertEquals("6", cSVFormat1.getRecordSeparator());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      QuoteMode quoteMode0 = QuoteMode.NONE;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      cSVFormat0.hashCode();
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("\r\n");
      cSVFormat1.hashCode();
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('6');
      cSVFormat0.hashCode();
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
      assertEquals('6', cSVFormat0.getDelimiter());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      Character character0 = Character.valueOf('x');
      CSVFormat cSVFormat1 = cSVFormat0.DEFAULT.withCommentMarker(character0);
      String string0 = cSVFormat1.toString();
      assertEquals("Delimiter=<,> QuoteChar=<\"> CommentStart=<x> RecordSeparator=<\r\n> EmptyLines:ignored SkipHeaderRecord:false", string0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      String string0 = cSVFormat0.toString();
      assertEquals("Delimiter=<\t> Escape=<\\> RecordSeparator=<\n> SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("");
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat2 = cSVFormat1.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      String string0 = cSVFormat2.toString();
      assertEquals("Delimiter=<,> QuoteChar=<\"> NullString=<> RecordSeparator=<\r\n> EmptyLines:ignored SkipHeaderRecord:false Header:[]", string0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('T');
      String string0 = cSVFormat0.toString();
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertEquals("Delimiter=<T> SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreHeaderCase();
      String string0 = cSVFormat1.toString();
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals("Delimiter=<\t> QuoteChar=<\"> RecordSeparator=<\r\n> EmptyLines:ignored SurroundingSpaces:ignored IgnoreHeaderCase:ignored SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Object[] objectArray0 = new Object[3];
      CSVFormat cSVFormat1 = cSVFormat0.withHeaderComments(objectArray0);
      String string0 = cSVFormat1.toString();
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals("Delimiter=<\t> QuoteChar=<\"> RecordSeparator=<\r\n> EmptyLines:ignored SurroundingSpaces:ignored SkipHeaderRecord:false HeaderComments:[null, null, null]", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      // Undeclared exception!
      try { 
        CSVFormat.newFormat('\n');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The delimiter cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      // Undeclared exception!
      try { 
        cSVFormat0.withQuote(',');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The quoteChar character and the delimiter cannot be the same (',')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      // Undeclared exception!
      try { 
        cSVFormat0.withCommentMarker(',');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start character and the delimiter cannot be the same (',')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      // Undeclared exception!
      try { 
        cSVFormat0.withCommentMarker('\"');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start character and the quoteChar cannot be the same ('\"')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      // Undeclared exception!
      try { 
        cSVFormat0.withCommentMarker('\\');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start and the escape character cannot be the same ('\\')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      QuoteMode quoteMode0 = QuoteMode.NONE;
      // Undeclared exception!
      try { 
        cSVFormat0.EXCEL.withQuoteMode(quoteMode0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No quotes mode set but no escape character is set
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      String[] stringArray0 = new String[8];
      // Undeclared exception!
      try { 
        cSVFormat0.withHeader(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The header contains a duplicate entry: 'null' in [null, null, null, null, null, null, null, null]
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      // Undeclared exception!
      try { 
        cSVFormat0.withCommentMarker('\r');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start marker character cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withDelimiter('i');
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals('i', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      // Undeclared exception!
      try { 
        cSVFormat0.withEscape('\r');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The escape character cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSet) null);
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn((ResultSetMetaData) null).when(resultSet0).getMetaData();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(resultSet0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      rowSetMetaDataImpl0.setColumnCount(824);
      // Undeclared exception!
      try { 
        cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The header contains a duplicate entry: 'null' in [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }
}

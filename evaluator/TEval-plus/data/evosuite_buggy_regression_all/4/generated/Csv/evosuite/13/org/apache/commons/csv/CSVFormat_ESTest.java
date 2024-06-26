/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:32:53 GMT 2023
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
import org.apache.commons.csv.QuoteMode;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVFormat_ESTest extends CSVFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[9];
      stringArray0[0] = "Escape=<";
      stringArray0[1] = "@";
      stringArray0[2] = "\"08:ityCc?K.=189}p";
      stringArray0[3] = " EmptyLines:ignored";
      stringArray0[4] = "Unexpected Quote value: ";
      stringArray0[5] = "EOF whilst processing escape sequence";
      stringArray0[6] = "org.apache.commons.csv.Constants";
      stringArray0[7] = "D6TF:vvdk5nICK";
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      cSVFormat1.format(stringArray0);
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.equals((Object)cSVFormat0));
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      // Undeclared exception!
      try { 
        cSVFormat0.print((Appendable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Parameter 'out' must not be null!
         //
         verifyException("org.apache.commons.csv.Assertions", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      QuoteMode quoteMode0 = QuoteMode.ALL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.DEFAULT.withIgnoreHeaderCase();
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        CSVFormat.valueOf("@?.+Se");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No enum constant org.apache.commons.csv.CSVFormat.Predefined.@?.+Se
         //
         verifyException("java.lang.Enum", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreEmptyLines();
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String string0 = cSVFormat0.TDF.toString();
      assertEquals("Delimiter=<\t> QuoteChar=<\"> RecordSeparator=<\r\n> EmptyLines:ignored SurroundingSpaces:ignored SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      PipedReader pipedReader0 = new PipedReader();
      CSVParser cSVParser0 = cSVFormat0.MYSQL.parse(pipedReader0);
      assertEquals(0L, cSVParser0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withQuote('f');
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals('f', (char)cSVFormat1.getQuoteCharacter());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat0.getAllowMissingColumnNames();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withRecordSeparator('t');
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertEquals("t", cSVFormat1.getRecordSeparator());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      // Undeclared exception!
      try { 
        cSVFormat0.withEscape('\n');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The escape character cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      // Undeclared exception!
      try { 
        cSVFormat0.withQuote('\r');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The quoteChar cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker((Character) null);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      Object[] objectArray0 = new Object[6];
      objectArray0[2] = (Object) cSVFormat0;
      CSVFormat cSVFormat1 = cSVFormat0.withHeaderComments(objectArray0);
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      boolean boolean0 = cSVFormat0.equals(cSVFormat0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      boolean boolean0 = cSVFormat0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Object object0 = new Object();
      boolean boolean0 = cSVFormat0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withDelimiter('G');
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(boolean0);
      assertFalse(cSVFormat0.equals((Object)cSVFormat1));
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertEquals('G', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Character character0 = new Character('f');
      CSVFormat cSVFormat1 = cSVFormat0.withEscape(character0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.equals((Object)cSVFormat0));
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals('\"', (char)cSVFormat1.getQuoteCharacter());
      assertFalse(boolean0);
      assertEquals('f', (char)cSVFormat1.getEscapeCharacter());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals('\t', cSVFormat1.getDelimiter());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("");
      cSVFormat1.equals(cSVFormat0);
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertEquals("\n", cSVFormat1.getRecordSeparator());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals("", cSVFormat1.getNullString());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Character character0 = new Character('+');
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker(character0);
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('K');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertEquals('K', (char)cSVFormat1.getCommentMarker());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Character character0 = new Character('+');
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker(character0);
      CSVFormat cSVFormat2 = cSVFormat1.withRecordSeparator("org.apache.commons.csv.Lexer");
      boolean boolean0 = cSVFormat1.equals(cSVFormat2);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat2.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat2.isNullStringSet());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat2.isQuoteCharacterSet());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      Character character0 = Character.valueOf('n');
      CSVFormat cSVFormat1 = cSVFormat0.withEscape(character0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("t");
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals("t", cSVFormat1.getNullString());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
      assertEquals("\n", cSVFormat1.getRecordSeparator());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.RFC4180.withNullString("");
      CSVFormat cSVFormat2 = cSVFormat1.withIgnoreSurroundingSpaces(false);
      boolean boolean0 = cSVFormat2.equals(cSVFormat1);
      assertEquals("", cSVFormat2.getNullString());
      assertFalse(cSVFormat2.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals("\r\n", cSVFormat2.getRecordSeparator());
      assertTrue(boolean0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat2.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreSurroundingSpaces();
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertEquals(',', cSVFormat1.getDelimiter());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat.Predefined cSVFormat_Predefined0 = CSVFormat.Predefined.Default;
      CSVFormat cSVFormat1 = cSVFormat_Predefined0.getFormat();
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withSkipHeaderRecord();
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('X');
      CSVFormat cSVFormat1 = cSVFormat0.withRecordSeparator("uqh,zB");
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals('X', cSVFormat0.getDelimiter());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertFalse(cSVFormat0.getSkipHeaderRecord());
      assertFalse(cSVFormat1.isNullStringSet());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat(':');
      CSVFormat cSVFormat1 = cSVFormat0.withAllowMissingColumnNames();
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertTrue(boolean0);
      assertTrue(cSVFormat1.getAllowMissingColumnNames());
      assertEquals(':', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Object[] objectArray0 = new Object[9];
      CSVFormat cSVFormat1 = cSVFormat0.MYSQL.withHeaderComments(objectArray0);
      cSVFormat1.getHeaderComments();
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      QuoteMode quoteMode0 = QuoteMode.ALL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      cSVFormat0.hashCode();
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('d');
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.RFC4180.withNullString("");
      CSVFormat cSVFormat2 = cSVFormat1.withIgnoreSurroundingSpaces();
      cSVFormat2.hashCode();
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat2.isEscapeCharacterSet());
      assertEquals("", cSVFormat2.getNullString());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat2.getIgnoreSurroundingSpaces());
      assertEquals("\r\n", cSVFormat2.getRecordSeparator());
      assertFalse(cSVFormat2.getSkipHeaderRecord());
      assertFalse(cSVFormat2.getAllowMissingColumnNames());
      assertEquals(',', cSVFormat2.getDelimiter());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = cSVFormat0.withSkipHeaderRecord(true);
      cSVFormat1.hashCode();
      assertTrue(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat(':');
      cSVFormat0.hashCode();
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
      assertFalse(cSVFormat0.getIgnoreHeaderCase());
      assertFalse(cSVFormat0.getSkipHeaderRecord());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CSVFormat.Predefined cSVFormat_Predefined0 = CSVFormat.Predefined.MySQL;
      CSVFormat cSVFormat0 = cSVFormat_Predefined0.getFormat();
      Character character0 = new Character('z');
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker(character0);
      String string0 = cSVFormat1.toString();
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals("Delimiter=<\t> Escape=<\\> CommentStart=<z> RecordSeparator=<\n> SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CSVFormat.Predefined cSVFormat_Predefined0 = CSVFormat.Predefined.MySQL;
      CSVFormat cSVFormat0 = cSVFormat_Predefined0.getFormat();
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("org.apache.commons.csv.CSVParser");
      String string0 = cSVFormat1.toString();
      assertEquals("Delimiter=<\t> Escape=<\\> NullString=<org.apache.commons.csv.CSVParser> RecordSeparator=<\n> SkipHeaderRecord:false", string0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('A');
      String string0 = cSVFormat0.toString();
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertEquals("Delimiter=<A> SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.DEFAULT.withIgnoreHeaderCase();
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      CSVFormat cSVFormat2 = cSVFormat1.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
      String string0 = cSVFormat2.toString();
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertTrue(cSVFormat1.getIgnoreHeaderCase());
      assertEquals("Delimiter=<,> QuoteChar=<\"> RecordSeparator=<\r\n> EmptyLines:ignored IgnoreHeaderCase:ignored SkipHeaderRecord:false Header:[]", string0);
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      Object[] objectArray0 = new Object[9];
      CSVFormat cSVFormat1 = cSVFormat0.withHeaderComments(objectArray0);
      String string0 = cSVFormat1.toString();
      assertEquals("Delimiter=<,> QuoteChar=<\"> RecordSeparator=<\r\n> EmptyLines:ignored SkipHeaderRecord:false HeaderComments:[null, null, null, null, null, null, null, null, null]", string0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      // Undeclared exception!
      try { 
        cSVFormat0.withDelimiter('\"');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The quoteChar character and the delimiter cannot be the same ('\"')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      // Undeclared exception!
      try { 
        cSVFormat0.withDelimiter('\\');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The escape character and the delimiter cannot be the same ('\\')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
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
  public void test50()  throws Throwable  {
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
  public void test51()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape('6');
      // Undeclared exception!
      try { 
        cSVFormat1.withCommentMarker('6');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start and the escape character cannot be the same ('6')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      QuoteMode quoteMode0 = QuoteMode.NONE;
      // Undeclared exception!
      try { 
        cSVFormat0.withQuoteMode(quoteMode0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No quotes mode set but no escape character is set
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      RowSetMetaDataImpl rowSetMetaDataImpl0 = new RowSetMetaDataImpl();
      rowSetMetaDataImpl0.setColumnCount(4);
      // Undeclared exception!
      try { 
        cSVFormat0.withHeader((ResultSetMetaData) rowSetMetaDataImpl0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The header contains a duplicate entry: 'null' in [null, null, null, null]
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      // Undeclared exception!
      try { 
        cSVFormat0.withCommentMarker('\n');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start marker character cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
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
  public void test56()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.DEFAULT.withHeader((ResultSet) null);
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertEquals(',', cSVFormat1.getDelimiter());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      ResultSet resultSet0 = mock(ResultSet.class, new ViolatedAssumptionAnswer());
      doReturn((ResultSetMetaData) null).when(resultSet0).getMetaData();
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(resultSet0);
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
      assertFalse(cSVFormat1.getIgnoreHeaderCase());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }
}

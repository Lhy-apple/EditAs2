/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:32:18 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.PipedReader;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.QuoteMode;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVFormat_ESTest extends CSVFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      Object[] objectArray0 = new Object[2];
      String string0 = cSVFormat0.RFC4180.format(objectArray0);
      assertEquals("\"\",", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      boolean boolean0 = cSVFormat0.getSkipHeaderRecord();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("EORECORD");
      CSVPrinter cSVPrinter0 = cSVFormat0.print(mockPrintWriter0);
      assertNotNull(cSVPrinter0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withSkipHeaderRecord(true);
      cSVFormat1.hashCode();
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.getSkipHeaderRecord());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        CSVFormat.newFormat('\r');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The delimiter cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withIgnoreEmptyLines(true);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertTrue(boolean0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[2];
      // Undeclared exception!
      try { 
        cSVFormat0.withHeader(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The header contains a duplicate entry: 'null' in [null, null]
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      PipedReader pipedReader0 = new PipedReader();
      CSVParser cSVParser0 = cSVFormat0.TDF.parse(pipedReader0);
      assertFalse(cSVParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("Qjt1KGPk;n|[B/C");
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertEquals("\r\n", cSVFormat1.getRecordSeparator());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertEquals("Qjt1KGPk;n|[B/C", cSVFormat1.getNullString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      boolean boolean0 = cSVFormat0.getAllowMissingColumnNames();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape('-');
      // Undeclared exception!
      try { 
        cSVFormat1.withCommentMarker('-');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start and the escape character cannot be the same ('-')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withAllowMissingColumnNames(false);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('x');
      CSVFormat cSVFormat1 = cSVFormat0.withRecordSeparator('x');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals('x', cSVFormat0.getDelimiter());
      assertFalse(boolean0);
      assertEquals("x", cSVFormat1.getRecordSeparator());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Character character0 = new Character('\n');
      // Undeclared exception!
      try { 
        cSVFormat0.withCommentMarker(character0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start marker character cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker((Character) null);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[2];
      stringArray0[0] = "The comment start marker character cannot be a line break";
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      String string0 = cSVFormat1.toString();
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals("Delimiter=<\t> QuoteChar=<\"> RecordSeparator=<\r\n> EmptyLines:ignored SurroundingSpaces:ignored SkipHeaderRecord:false Header:[The comment start marker character cannot be a line break, null]", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      Character character0 = new Character('N');
      CSVFormat cSVFormat1 = cSVFormat0.withQuote(character0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals('N', (char)cSVFormat1.getQuoteCharacter());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(boolean0);
      assertEquals('\\', (char)cSVFormat1.getEscapeCharacter());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat0.equals(cSVFormat0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      boolean boolean0 = cSVFormat0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      boolean boolean0 = cSVFormat0.equals("out");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = CSVFormat.DEFAULT;
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      QuoteMode quoteMode0 = QuoteMode.MINIMAL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withSkipHeaderRecord(true);
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuote('O');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals('O', (char)cSVFormat1.getQuoteCharacter());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('6');
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertEquals('6', (char)cSVFormat1.getCommentMarker());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('k');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals('k', (char)cSVFormat1.getCommentMarker());
      assertFalse(boolean0);
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Character character0 = Character.valueOf('Y');
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker(character0);
      CSVFormat cSVFormat2 = cSVFormat1.withIgnoreSurroundingSpaces(false);
      boolean boolean0 = cSVFormat2.equals(cSVFormat1);
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat2.isNullStringSet());
      assertFalse(cSVFormat2.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat2.isQuoteCharacterSet());
      assertTrue(cSVFormat2.getIgnoreEmptyLines());
      assertFalse(cSVFormat2.isEscapeCharacterSet());
      assertFalse(cSVFormat1.equals((Object)cSVFormat0));
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape('=');
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals('\"', (char)cSVFormat1.getQuoteCharacter());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertEquals('=', (char)cSVFormat1.getEscapeCharacter());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVFormat cSVFormat1 = cSVFormat0.withEscape('-');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertEquals('\"', (char)cSVFormat1.getQuoteCharacter());
      assertEquals('-', (char)cSVFormat1.getEscapeCharacter());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("\r\n");
      boolean boolean0 = cSVFormat1.equals(cSVFormat0);
      assertFalse(boolean0);
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertEquals("\r\n", cSVFormat1.getNullString());
      assertEquals("\n", cSVFormat1.getRecordSeparator());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertEquals('\t', cSVFormat1.getDelimiter());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("O_g>{}r\"Lu?");
      CSVFormat cSVFormat2 = cSVFormat1.withRecordSeparator("O_g>{}r\"Lu?");
      boolean boolean0 = cSVFormat1.equals(cSVFormat2);
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat2.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(boolean0);
      assertEquals("O_g>{}r\"Lu?", cSVFormat1.getNullString());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat2.isQuoteCharacterSet());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      String[] stringArray0 = new String[0];
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(boolean0);
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVFormat cSVFormat1 = CSVFormat.DEFAULT;
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('x');
      CSVFormat cSVFormat1 = CSVFormat.newFormat('x');
      boolean boolean0 = cSVFormat0.equals(cSVFormat1);
      assertTrue(boolean0);
      assertFalse(cSVFormat1.getSkipHeaderRecord());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertEquals('x', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      String[] stringArray0 = new String[4];
      stringArray0[0] = "\r\n";
      stringArray0[1] = "l2-b#6$na'B0";
      stringArray0[2] = "QuoteChar=<";
      CSVFormat cSVFormat1 = cSVFormat0.withHeader(stringArray0);
      cSVFormat1.getHeader();
      assertFalse(cSVFormat1.isEscapeCharacterSet());
      assertEquals(',', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.isNullStringSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      QuoteMode quoteMode0 = QuoteMode.MINIMAL;
      CSVFormat cSVFormat1 = cSVFormat0.withQuoteMode(quoteMode0);
      cSVFormat1.hashCode();
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('^');
      cSVFormat1.hashCode();
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isQuoteCharacterSet());
      assertTrue(cSVFormat1.isEscapeCharacterSet());
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("Qjt1KGPk;n|[B/C");
      cSVFormat1.hashCode();
      assertEquals('\t', cSVFormat1.getDelimiter());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('1');
      cSVFormat0.hashCode();
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertFalse(cSVFormat0.getIgnoreSurroundingSpaces());
      assertEquals('1', cSVFormat0.getDelimiter());
      assertFalse(cSVFormat0.getIgnoreEmptyLines());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      Character character0 = Character.valueOf('Y');
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker(character0);
      String string0 = cSVFormat1.toString();
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertEquals("Delimiter=<\t> QuoteChar=<\"> CommentStart=<Y> RecordSeparator=<\r\n> EmptyLines:ignored SurroundingSpaces:ignored SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withNullString("c");
      CSVFormat cSVFormat2 = cSVFormat1.withRecordSeparator('B');
      String string0 = cSVFormat2.toString();
      assertFalse(cSVFormat1.getIgnoreEmptyLines());
      assertEquals("Delimiter=<\t> Escape=<\\> NullString=<c> RecordSeparator=<B> SkipHeaderRecord:false", string0);
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertFalse(cSVFormat1.getIgnoreSurroundingSpaces());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('u');
      String string0 = cSVFormat0.toString();
      assertFalse(cSVFormat0.getAllowMissingColumnNames());
      assertEquals("Delimiter=<u> SkipHeaderRecord:false", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('x');
      // Undeclared exception!
      try { 
        cSVFormat0.withQuote('x');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The quoteChar character and the delimiter cannot be the same ('x')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.newFormat('u');
      // Undeclared exception!
      try { 
        cSVFormat0.withEscape('u');
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The escape character and the delimiter cannot be the same ('u')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
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
  public void test45()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVFormat cSVFormat1 = cSVFormat0.withCommentMarker('^');
      Character character0 = Character.valueOf('^');
      // Undeclared exception!
      try { 
        cSVFormat1.withQuote(character0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The comment start character and the quoteChar cannot be the same ('^')
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
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
  public void test47()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVFormat cSVFormat1 = cSVFormat0.withDelimiter('F');
      assertEquals('F', cSVFormat1.getDelimiter());
      assertFalse(cSVFormat1.isNullStringSet());
      assertTrue(cSVFormat1.isQuoteCharacterSet());
      assertFalse(cSVFormat1.getAllowMissingColumnNames());
      assertTrue(cSVFormat1.getIgnoreEmptyLines());
      assertTrue(cSVFormat1.getIgnoreSurroundingSpaces());
      assertFalse(cSVFormat1.isEscapeCharacterSet());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
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
  public void test49()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
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
  public void test50()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      Character character0 = new Character('\n');
      // Undeclared exception!
      try { 
        cSVFormat0.withQuote(character0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The quoteChar cannot be a line break
         //
         verifyException("org.apache.commons.csv.CSVFormat", e);
      }
  }
}

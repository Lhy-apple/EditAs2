/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:51:06 GMT 2023
 */

package org.jsoup.select;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.ListIterator;
import java.util.Set;
import java.util.regex.Pattern;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeVisitor;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Elements_ESTest extends Elements_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Elements elements0 = new Elements();
      Document[] documentArray0 = new Document[0];
      Document[] documentArray1 = elements0.toArray(documentArray0);
      assertEquals(0, documentArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.remove(330);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 330, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      boolean boolean0 = elements0.removeAll(elements0);
      assertEquals(0, elements0.size());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.add(1, (Element) null);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      boolean boolean0 = elements0.equals(document0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.listIterator(10731);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 10731
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Elements elements0 = new Elements();
      boolean boolean0 = elements0.remove((Object) elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Elements elements0 = new Elements();
      boolean boolean0 = elements0.is("V4dWd~");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Elements elements0 = new Elements();
      elements0.clear();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Element[] elementArray0 = new Element[5];
      Elements elements0 = new Elements(elementArray0);
      NodeVisitor nodeVisitor0 = mock(NodeVisitor.class, new ViolatedAssumptionAnswer());
      Elements elements1 = elements0.traverse(nodeVisitor0);
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Elements elements0 = new Elements();
      int int0 = elements0.indexOf(elements0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Elements elements0 = new Elements();
      Object object0 = new Object();
      int int0 = elements0.lastIndexOf(object0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.not("}Y&AEm");
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Elements elements0 = new Elements();
      elements0.hashCode();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      String string0 = elements0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.addAll((-6), (Collection<? extends Element>) elements0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -6, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Elements elements0 = new Elements();
      ListIterator<Element> listIterator0 = elements0.listIterator();
      assertFalse(listIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.subList((-392), (-392));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // fromIndex = -392
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Elements elements0 = new Elements();
      boolean boolean0 = elements0.addAll((Collection<? extends Element>) elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.set((-5588), (Element) null);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Elements elements0 = new Elements();
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      boolean boolean0 = elements0.retainAll(linkedHashSet0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.clone();
      assertNotSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Element[] elementArray0 = new Element[1];
      Elements elements0 = new Elements(elementArray0);
      // Undeclared exception!
      try { 
        elements0.clone();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.select.Elements", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = Document.createShell("*Y");
      Elements elements0 = document0.getElementsByIndexLessThan(1375);
      String string0 = elements0.attr(";'ay>B'-2Io\"T");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      elements0.attr("org.jsoup.select.Elements", "");
      String string0 = elements0.attr("org.jsoup.select.Elements");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      boolean boolean0 = elements0.hasAttr("org.jsoup.select.Elements");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      elements0.attr(" ", "\n");
      boolean boolean0 = elements0.hasAttr("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = Document.createShell("fpartint");
      Elements elements0 = document0.getElementsContainingText("");
      Elements elements1 = elements0.removeAttr("fpartint");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsContainingOwnText("");
      Elements elements1 = elements0.addClass("");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      Elements elements1 = elements0.removeClass("");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = Document.createShell("X");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.toggleClass("X");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = Document.createShell("*Y");
      Elements elements0 = document0.getElementsByIndexLessThan(1379);
      boolean boolean0 = elements0.hasClass("*Y");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("m$c{+`tvD#/)E>!\"da");
      Pattern pattern0 = Pattern.compile("");
      Elements elements0 = document0.getElementsMatchingOwnText(pattern0);
      boolean boolean0 = elements0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Elements elements0 = new Elements();
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = Document.createShell("YLs!Oy)swy68X|/;g?");
      Elements elements0 = document0.getElementsByAttributeValueNot("(=#0apS2d&_&U@", "7K=,!M!");
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = Document.createShell("");
      Pattern pattern0 = Pattern.compile("");
      Elements elements0 = document0.getElementsMatchingText(pattern0);
      Elements elements1 = elements0.val("O");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("Enqv6");
      Element[] elementArray0 = new Element[2];
      elementArray0[0] = (Element) document0;
      elementArray0[1] = (Element) document0;
      Elements elements0 = new Elements(elementArray0);
      elements0.prepend("Enqv6");
      String string0 = elements0.text();
      assertEquals("Enqv6Enqv6 Enqv6Enqv6", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("l");
      Elements elements0 = document0.children();
      elements0.add((Element) document0);
      boolean boolean0 = elements0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = new Elements();
      elements0.add((Element) document0);
      Elements elements1 = elements0.html("*Y");
      boolean boolean0 = elements1.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getElementsContainingOwnText("");
      String string0 = elements0.html();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>\n\n", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = Document.createShell("*Y");
      Elements elements0 = document0.getElementsByIndexLessThan(1379);
      String string0 = elements0.outerHtml();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = Document.createShell("X");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.tagName("X");
      boolean boolean0 = elements1.is("X");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = Document.createShell("Lopf");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.append("Lopf");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.before((String) null);
      assertEquals(0, elements1.size());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("(yOE_%>");
      Elements elements0 = document0.getElementsContainingOwnText("");
      // Undeclared exception!
      try { 
        elements0.before("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.after("");
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("perp");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-2082));
      // Undeclared exception!
      try { 
        elements0.after("perp");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.wrap("I,oLxt");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      // Undeclared exception!
      try { 
        elements0.wrap("org.jsoup.select.Elements");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("l");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.unwrap();
      assertEquals(0, elements1.size());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("l");
      Elements elements0 = document0.getElementsMatchingText("");
      // Undeclared exception!
      try { 
        elements0.unwrap();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      Elements elements1 = elements0.empty();
      assertEquals(1, elements1.size());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("l");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.remove();
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("Star");
      Elements elements0 = document0.getElementsMatchingOwnText("");
      // Undeclared exception!
      try { 
        elements0.remove();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("l");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.eq(0);
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.eq((-4351));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = Document.createShell("*Y");
      Elements elements0 = document0.getElementsByIndexLessThan(1379);
      Elements elements1 = elements0.parents();
      assertEquals(1, elements1.size());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Elements elements0 = new Elements();
      Element element0 = elements0.first();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      Element element0 = elements0.last();
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Elements elements0 = new Elements();
      Element element0 = elements0.last();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Elements elements0 = new Elements();
      boolean boolean0 = elements0.contains("C");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Elements elements0 = new Elements();
      Document document0 = new Document("H");
      Set<String> set0 = document0.classNames();
      boolean boolean0 = elements0.containsAll(set0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      LinkedHashSet<Elements> linkedHashSet0 = new LinkedHashSet<Elements>();
      boolean boolean0 = elements0.containsAll(linkedHashSet0);
      assertTrue(boolean0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:05:54 GMT 2023
 */

package org.jsoup.select;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import java.util.LinkedList;
import java.util.function.UnaryOperator;
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
      boolean boolean0 = elements0.addAll((Collection<? extends Element>) elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.toArray((Elements[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.remove(0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("?}_$?);+rVQ2Xz3n");
      Elements elements0 = document0.getElementsByAttributeValueContaining("cL0Y6&X=j", "Ss'Lcs");
      boolean boolean0 = elements0.removeAll(elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Elements elements0 = new Elements();
      Document document0 = Document.createShell(" ");
      elements0.add(0, (Element) document0);
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingText("org.jsoup.select.Elements");
      boolean boolean0 = elements0.equals(document0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.listIterator(175);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 175
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Elements elements0 = new Elements();
      Object object0 = new Object();
      boolean boolean0 = elements0.remove(object0);
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
      Element[] elementArray0 = new Element[0];
      Elements elements0 = new Elements(elementArray0);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getElementsByIndexGreaterThan(881);
      int int0 = elements0.indexOf(document0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Elements elements0 = new Elements();
      int int0 = elements0.lastIndexOf("auFTwPtzXq`EQ}");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.not("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getAllElements();
      elements0.hashCode();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.addAll((-2572), (Collection<? extends Element>) elements0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -2572, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Elements elements0 = new Elements();
      UnaryOperator<Element> unaryOperator0 = UnaryOperator.identity();
      elements0.replaceAll(unaryOperator0);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getElementsByAttributeStarting("cL0Y6&X=j");
      // Undeclared exception!
      try { 
        elements0.subList((-1), (-1));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // fromIndex = -1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Elements elements0 = new Elements();
      boolean boolean0 = elements0.is("html");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getElementsByIndexLessThan(1451);
      // Undeclared exception!
      try { 
        elements0.set(1451, (Element) document0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1451, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Elements elements0 = new Elements();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      boolean boolean0 = elements0.retainAll(linkedList0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingOwnText("");
      Elements elements1 = elements0.clone();
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = Document.createShell("rhD");
      Elements elements0 = document0.getElementsByIndexLessThan(1);
      String string0 = elements0.attr(" ");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = Document.createShell("rhD");
      Elements elements0 = document0.getElementsByIndexLessThan(1);
      Elements elements1 = elements0.attr("rhD", "tUKlV~lJZLC");
      String string0 = elements1.attr("rhD");
      assertEquals("tUKlV~lJZLC", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasAttr("M<iM^Q=+<|");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getAllElements();
      elements0.attr("\n", "");
      boolean boolean0 = elements0.hasAttr("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.removeAttr("[;I]3Ax9i 3;");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = Document.createShell(":bE:/|MlSHU^79J(lp");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1850));
      Elements elements1 = elements0.addClass(":bE:/|MlSHU^79J(lp");
      assertEquals(4, elements1.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.removeClass("");
      assertEquals(1, elements1.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      Elements elements1 = elements0.toggleClass("");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = Document.createShell(":bE:/|MlSHU^79J(lp");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1850));
      boolean boolean0 = elements0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasClass("cL0Y6&X=j");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("cL0Y&X=j");
      Elements elements0 = document0.getElementsByAttributeStarting("cL0Y&X=j");
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = Document.createShell("[;I]3Ax9i 3;");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-535));
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getElementsMatchingText("");
      Elements elements1 = elements0.val("G<BT]NO\"-./C!RT$_");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = Document.createShell("boxUl");
      document0.text("subnE");
      Elements elements0 = document0.getElementsByIndexLessThan(1);
      String string0 = elements0.text();
      assertEquals("subnE subnE ", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingText("org.jsoup.select.Elements");
      elements0.add((Element) document0);
      boolean boolean0 = elements0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingOwnText("");
      document0.title("0v)");
      boolean boolean0 = elements0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getElementsMatchingText("");
      String string0 = elements0.html();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>\n\n", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Evaluator$ContainsText");
      Elements elements0 = document0.children();
      String string0 = elements0.toString();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = Document.createShell(" ");
      Elements elements0 = document0.getElementsByIndexLessThan(1203);
      String string0 = elements0.outerHtml();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.tagName("[;I]3Ax9i 3;");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.html("G");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingOwnText("");
      Elements elements1 = elements0.prepend("{Y% $%sak7A$D");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.append("@/-@iFzDc|^");
      assertEquals(1, elements1.size());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.before("Rz<OGx&0)>(]o+");
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("cL0Y6&X=j");
      Elements elements0 = document0.getElementsByIndexLessThan(124);
      // Undeclared exception!
      try { 
        elements0.before("cL0Y6&X=j");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.after("{Y% $%sak7A$D");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingText("org.jsoup.select.Elements");
      elements0.add((Element) document0);
      // Undeclared exception!
      try { 
        elements0.after(":!A>WJ~gIr-?H");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Evaluator$ContainsText");
      Elements elements0 = document0.getElementsByAttributeStarting("JF{;j");
      Elements elements1 = elements0.wrap("L<SK1{X#");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingOwnText("");
      // Undeclared exception!
      try { 
        elements0.wrap("%,u7T.1*?M");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.unwrap();
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getAllElements();
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
  public void test52()  throws Throwable  {
      Document document0 = Document.createShell("rhD");
      Elements elements0 = document0.getElementsByIndexLessThan(1);
      Elements elements1 = elements0.empty();
      assertEquals(3, elements1.size());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.remove();
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getElementsContainingOwnText("");
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
  public void test55()  throws Throwable  {
      Document document0 = Document.createShell("hidden");
      Elements elements0 = document0.getElementsByIndexLessThan(2038);
      Elements elements1 = elements0.eq(290);
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.eq((-3488));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = Document.createShell("rhD");
      Elements elements0 = document0.getElementsByIndexLessThan(1);
      boolean boolean0 = elements0.is("html");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Evaluator$ContainsText");
      Elements elements0 = document0.children();
      Elements elements1 = elements0.parents();
      assertNotSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getElementsMatchingText("");
      Document document1 = (Document)elements0.last();
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Elements elements0 = new Elements();
      Element element0 = elements0.last();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.select.Elements");
      Elements elements0 = document0.getElementsContainingOwnText("");
      NodeVisitor nodeVisitor0 = mock(NodeVisitor.class, new ViolatedAssumptionAnswer());
      Elements elements1 = elements0.traverse(nodeVisitor0);
      assertEquals(4, elements1.size());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Elements elements0 = new Elements();
      Object object0 = new Object();
      boolean boolean0 = elements0.contains(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Document document0 = new Document("[;I]3Ax9i 3;");
      Elements elements0 = document0.getElementsMatchingText("");
      boolean boolean0 = elements0.contains(document0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Elements elements0 = new Elements();
      Document document0 = Document.createShell("rhD");
      Elements elements1 = document0.getElementsByIndexLessThan(1);
      boolean boolean0 = elements0.containsAll(elements1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Elements elements0 = new Elements();
      LinkedList<Document> linkedList0 = new LinkedList<Document>();
      boolean boolean0 = elements0.containsAll(linkedList0);
      assertTrue(boolean0);
  }
}

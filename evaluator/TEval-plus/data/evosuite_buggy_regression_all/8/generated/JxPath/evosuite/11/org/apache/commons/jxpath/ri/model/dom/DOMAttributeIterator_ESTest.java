/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:41:54 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.dom;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.VariablePointer;
import org.apache.commons.jxpath.ri.model.dom.DOMAttributeIterator;
import org.apache.html.dom.HTMLDocumentImpl;
import org.apache.html.dom.HTMLHeadingElementImpl;
import org.apache.html.dom.HTMLParagraphElementImpl;
import org.apache.xerces.dom.CDATASectionImpl;
import org.apache.xerces.dom.PSVIDocumentImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DOMAttributeIterator_ESTest extends DOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      QName qName0 = new QName("9");
      PSVIDocumentImpl pSVIDocumentImpl0 = new PSVIDocumentImpl();
      CDATASectionImpl cDATASectionImpl0 = new CDATASectionImpl(pSVIDocumentImpl0, "9");
      Locale locale0 = Locale.ENGLISH;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, cDATASectionImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      int int0 = dOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLParagraphElementImpl hTMLParagraphElementImpl0 = new HTMLParagraphElementImpl(hTMLDocumentImpl0, "xmlns");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLParagraphElementImpl0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) hTMLParagraphElementImpl0);
      NodePointer nodePointer1 = nodePointer0.createAttribute(jXPathContext0, qName0);
      assertFalse(nodePointer1.isCollection());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLParagraphElementImpl hTMLParagraphElementImpl0 = new HTMLParagraphElementImpl(hTMLDocumentImpl0, "xmlns");
      hTMLParagraphElementImpl0.setClassName("xmlns");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLParagraphElementImpl0);
      QName qName1 = new QName("xmlns", "*");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, variablePointer0);
      HTMLHeadingElementImpl hTMLHeadingElementImpl0 = new HTMLHeadingElementImpl(hTMLDocumentImpl0, "<<unknown namespace>>");
      hTMLHeadingElementImpl0.setAttribute("xmlns", "<<unknown namespace>>");
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(nodePointer0, qName0, hTMLHeadingElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer1, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      QName qName0 = new QName("xmlns", "xmlns");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLParagraphElementImpl hTMLParagraphElementImpl0 = new HTMLParagraphElementImpl(hTMLDocumentImpl0, "xmlns");
      hTMLParagraphElementImpl0.setClassName("<<unknown namespace>>");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLParagraphElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("*");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLParagraphElementImpl hTMLParagraphElementImpl0 = new HTMLParagraphElementImpl(hTMLDocumentImpl0, "<<unknown namespace>>");
      hTMLParagraphElementImpl0.setClassName("<<unknown namespace>>");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLParagraphElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("h");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLParagraphElementImpl hTMLParagraphElementImpl0 = new HTMLParagraphElementImpl(hTMLDocumentImpl0, "h");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLParagraphElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLDocumentImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
      
      dOMAttributeIterator0.getNodePointer();
      assertEquals(1, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLDocumentImpl0);
      HTMLHeadingElementImpl hTMLHeadingElementImpl0 = new HTMLHeadingElementImpl(hTMLDocumentImpl0, "<<unknown namespace>>");
      NodePointer nodePointer1 = NodePointer.newChildNodePointer(nodePointer0, qName0, hTMLHeadingElementImpl0);
      hTMLHeadingElementImpl0.setLang("<<unknown namespace>>");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer1, qName0);
      NodePointer nodePointer2 = dOMAttributeIterator0.getNodePointer();
      assertEquals(0, dOMAttributeIterator0.getPosition());
      assertNotNull(nodePointer2);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      QName qName0 = new QName("h", "h");
      VariablePointer variablePointer0 = new VariablePointer(qName0);
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLParagraphElementImpl hTMLParagraphElementImpl0 = new HTMLParagraphElementImpl(hTMLDocumentImpl0, "h");
      NodePointer nodePointer0 = NodePointer.newChildNodePointer(variablePointer0, qName0, hTMLParagraphElementImpl0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = dOMAttributeIterator0.setPosition((-2106));
      assertEquals((-2106), dOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}